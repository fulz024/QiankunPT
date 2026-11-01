"""Distributed semistochastic EN-PT2 with a serializable statistical report."""
from dataclasses import asdict, dataclass
import math
import time
import numpy as np
import torch
import torch.distributed as dist
from . import __version__
from .policy import choose_fixed_sources, single_sketch_bucket_budget
from .statistics import control_variance_bound, estimate


@dataclass(frozen=True)
class PT2Config:
    fine_threshold: float = 5e-8
    coarse_threshold: float = 5e-7
    target_standard_error: float = 1e-5
    unique_sources: int = 262144
    external_partitions: int = 16
    minimum_windows: int = 6
    maximum_windows: int = 10000
    maximum_seconds: float = 86400
    seed: int = 170000213
    memory_fraction: float = 0.9
    memory_gib: float | None = None
    safety_gib: float = 4.0
    maximum_draws: int | None = None
    buckets: int | None = None
    control_buckets: int | None = None
    source_batch_size: int = 8192
    denominator_epsilon: float = 1e-10

    def __post_init__(self):
        for name in ("fine_threshold", "coarse_threshold", "target_standard_error", "maximum_seconds",
                     "memory_fraction", "safety_gib", "denominator_epsilon"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.fine_threshold <= self.coarse_threshold:
            raise ValueError("Require 0 <= fine_threshold <= coarse_threshold")
        if min(self.target_standard_error, self.maximum_seconds, self.denominator_epsilon) <= 0:
            raise ValueError("SE, time limit, and denominator floor must be positive")
        for name in ("unique_sources", "external_partitions", "minimum_windows", "maximum_windows", "source_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.external_partitions & (self.external_partitions-1):
            raise ValueError("external_partitions must be a power of two")
        if not 2 <= self.minimum_windows <= self.maximum_windows:
            raise ValueError("Require 2 <= minimum_windows <= maximum_windows")
        if type(self.seed) is not int or not 0 <= self.seed < (1 << 63)-30_000_000_000:
            raise ValueError("seed outside the nonnegative signed 64-bit range reserved for stage streams")
        if not 0 < self.memory_fraction <= 1 or self.safety_gib < 0:
            raise ValueError("Invalid memory fraction or safety reserve")
        if self.memory_gib is not None and (not math.isfinite(self.memory_gib) or self.memory_gib <= 0):
            raise ValueError("memory_gib must be positive")
        if self.maximum_draws is not None and (type(self.maximum_draws) is not int or self.maximum_draws < 2):
            raise ValueError("maximum_draws must be an integer >= 2")
        for name in ("buckets", "control_buckets"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1 << 18 or value & (value-1)):
                raise ValueError(f"{name} must be a power of two >= 2**18")


@dataclass(frozen=True)
class PT2Result:
    report: dict

    @property
    def converged(self):
        """Whether the requested reported statistical SE was reached."""
        return self.report["status"] == "TARGET_REACHED"

    def to_dict(self):
        import copy
        return copy.deepcopy(self.report)


@torch.no_grad()
def evaluate(hamiltonian, reference, config=None, *, progress=None):
    """Evaluate external EN-PT2; this function neither selects nor solves CI.

    All ranks in an initialized NCCL group must call with identical references
    and configuration. Each rank holds the reference and Hamiltonian tables.
    progress(report) is called on rank zero after completed windows.
    """
    from ._backend.stochastic_resident import StochasticConfig, StochasticResidentPT2
    from ._backend.semistochastic_resident import SemistochasticResidentPT2
    cfg = config or PT2Config()
    ham = hamiltonian
    if reference.n_spin_orbitals != ham.n_orb:
        raise ValueError("Reference and Hamiltonian spin-orbital counts differ")
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if world > 1:
        if dist.get_backend() != "nccl":
            raise ValueError("Multi-GPU evaluation requires one process per GPU and NCCL")
        import hashlib
        digest = hashlib.sha256()
        digest.update(memoryview(reference.keys).cast("B"))
        digest.update(memoryview(reference.coefficients).cast("B"))
        fingerprint = digest.hexdigest()
        identities = [None] * world
        dist.all_gather_object(identities, (fingerprint, reference.energy, asdict(cfg)))
        if any(x != identities[0] for x in identities):
            raise ValueError("Ranks supplied different references or configurations")
    def minimum(value):
        t = torch.tensor(value, device=ham.device, dtype=torch.int64)
        if world > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return int(t.item())
    def any_rank(value):
        t = torch.tensor(int(value), device=ham.device, dtype=torch.int32)
        if world > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())
    torch.cuda.synchronize(ham.device)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(ham.device)
    base = StochasticResidentPT2(ham, reference.keys.copy(), reference.coefficients.copy(), reference.energy)
    total = torch.cuda.get_device_properties(ham.device).total_memory
    cap = minimum(min(int(total*cfg.memory_fraction), int(cfg.memory_gib*2**30) if cfg.memory_gib else total))
    # The draw workspace and reference/probability temporaries have explicit
    # room in addition to the stage-specific sketch fields.
    reserve = max(int(.1*cap), 24*len(base.keys)+(256 << 20))
    maximum_draws = (reserve-24*len(base.keys)-(64 << 20))//128
    maximum_draws = minimum(min(maximum_draws, cfg.maximum_draws) if cfg.maximum_draws else maximum_draws)
    if maximum_draws < 2:
        raise MemoryError("Insufficient source sampling workspace")
    source_plan = choose_fixed_sources(base.probability, cfg.unique_sources, maximum_draws)
    engine = SemistochasticResidentPT2(base, source_plan["deterministic_sources"])
    resident = torch.cuda.memory_allocated(ham.device)
    free, _ = torch.cuda.mem_get_info(ham.device)
    reclaimable = max(0, torch.cuda.memory_reserved(ham.device)-resident)
    available = minimum(int(min(cap-resident, free+reclaimable)-cfg.safety_gib*2**30))
    budget = single_sketch_bucket_budget(available-reserve, available)
    b = cfg.buckets or budget["tail_buckets"]
    cb = cfg.control_buckets or budget["control_buckets"]
    if b > budget["tail_buckets"] or cb > budget["control_buckets"]:
        raise MemoryError("Requested sketch exceeds the available stage memory budget")
    control_config = StochasticConfig(buckets=cb, source_batch_size=cfg.source_batch_size,
        stream_shards=4, diagonal_policy="incremental", denominator_epsilon=cfg.denominator_epsilon)
    tail_config = StochasticConfig(buckets=b, source_batch_size=cfg.source_batch_size,
        stream_shards=8, diagonal_policy="incremental", external_partitions=cfg.external_partitions,
        denominator_epsilon=cfg.denominator_epsilon)
    report = dict(schema_version=1, software="QiankunPT", version=__version__, status="RUNNING",
        reference_energy=reference.energy, support_size=len(reference.keys), n_spin_orbitals=ham.n_orb,
        coefficient_norm=float(base.norm.item()), config=asdict(cfg), world_size=world,
        device=torch.cuda.get_device_name(ham.device), torch_version=torch.__version__,
        frozen_settings=dict(source_plan, buckets=b, control_buckets=cb, partitions=cfg.external_partitions),
        memory=dict(cap_bytes=cap, resident_bytes=resident, source_reserve_bytes=reserve,
                    safety_bytes=int(cfg.safety_gib*2**30), **budget),
        control=[], tail=[], standard_error=None,
        uncertainty="Reported statistical SE: empirical independent-window tail variance plus a model-based control bound. Screening/regularization bias and coverage are not included.")
    report["setup_seconds"] = time.perf_counter()-start
    controls, norms, tails = [], [], []
    def finish(status):
        torch.cuda.synchronize(ham.device)
        if any_rank(torch.cuda.max_memory_allocated(ham.device) > cap):
            raise MemoryError("Measured allocated CUDA memory exceeds the requested cap")
        report.update(status=status, pt_seconds=time.perf_counter()-start,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(ham.device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(ham.device))
        if progress is not None and rank == 0:
            progress(PT2Result(report).to_dict())
        return PT2Result(report)
    def control_floor():
        return control_variance_bound(norms, cb)
    for window in range(16):
        if any_rank(time.perf_counter()-start > cfg.maximum_seconds):
            return finish("TIME_LIMIT")
        torch.cuda.synchronize(ham.device); before = time.perf_counter()
        fields, _, edges = engine._stream_fields([{}], [cfg.coarse_threshold], control_config,
            (cfg.seed+10_000_000_000+window*104729) & ((1 << 63)-1), 1)
        sectors = torch.linalg.vector_norm(fields[0,0].reshape(1,2,-1), dim=2).square()/base.norm
        controls.append(float((sectors[:,0]-sectors[:,1]).item()))
        norms.append(float(sectors.sum().item()))
        del fields, sectors
        torch.cuda.synchronize(ham.device)
        report["control"].append(dict(estimate=controls[-1], unsigned_norm=norms[-1],
                                      seconds=time.perf_counter()-before, external_edges=edges.tolist()))
        if not math.isfinite(controls[-1]) or not math.isfinite(norms[-1]):
            raise ArithmeticError("Nonfinite deterministic control estimate")
        if minimum(int(control_floor() < cfg.target_standard_error**2*.5)):
            break
    else:
        return finish("CONTROL_LIMIT")
    for window in range(cfg.maximum_windows):
        if any_rank(time.perf_counter()-start > cfg.maximum_seconds):
            return finish("TIME_LIMIT")
        torch.cuda.synchronize(ham.device); before = time.perf_counter()
        row = engine.tail(source_plan["draws"], cfg.fine_threshold,
            (cfg.seed+20_000_000_000+window*104729) & ((1 << 63)-1),
            coarse_threshold=cfg.coarse_threshold, config=tail_config)
        value = float(row["estimate"].item())
        if not math.isfinite(value):
            raise ArithmeticError("Nonfinite PT2 estimate")
        tails.append(value)
        torch.cuda.synchronize(ham.device)
        report["tail"].append(dict(estimate=value, seconds=time.perf_counter()-before,
            unique_sampled_sources=row["unique_sampled_sources"], external_edges=row["external_edges"].tolist()))
        if len(tails) >= 2:
            state = estimate(controls, tails, control_floor())
            report.update(state, total_energy=reference.energy+state["E2_external"])
            if minimum(int(len(tails) >= cfg.minimum_windows and state["standard_error"] <= cfg.target_standard_error)):
                return finish("TARGET_REACHED")
        if any_rank(torch.cuda.max_memory_allocated(ham.device) > cap):
            raise MemoryError("Measured allocated CUDA memory exceeds the requested cap")
        if progress is not None and rank == 0:
            progress(PT2Result(report).to_dict())
    return finish("MAX_WINDOWS")
