"""Distributed stochastic EN-PT2 with independently refreshed sketches.

Let x_i(a)=H_ai*c_i/sqrt(abs(d_a)) for external a, and Q be the
positive-sector inner product minus the negative-sector inner product.
For m independent draws I~p, Z=sum_k x_Ik/p_Ik. The estimator is

    (Q(SZ,SZ) - sum_k Q(x_Ik,x_Ik)/p_Ik**2) / (m*(m-1)*norm).

The default multinomial version replaces the within-source correction by
sum_i [n_i**2-n_i*(m-1)*p_i] Q(x_i,x_i)/p_i**2, estimating that diagonal
linearly in n_i instead of quadratically. Both versions have the same mean.

It is unbiased jointly over source sampling and ideal unbiased sketch maps,
relative to the fixed screened/regularized target. The implementation uses
seeded full-key SplitMix hashes. Exact diagnostic mode replaces Q(SZ,SZ)
with Q(Z,Z). Within-source self terms are contracted without sketching.

Independent windows refresh BOTH source draws and hash seeds. Replicas in one
window share source noise: their standard deviation is NOT a total error bar.
NCCL ranks divide the same global sampled sources and sum amplitudes before
squaring. Each rank holds the full reference support.
"""
from dataclasses import dataclass
import math

import torch
import torch.distributed as dist
from torch.profiler import record_function

from qiankunpt._backend.cuda_coupled import CudaHamiltonian


@dataclass(frozen=True)
class StochasticConfig:
    source_batch_size: int = 256
    max_edges: int = 8_000_000
    coefficient_threshold: float = 0.0
    denominator_epsilon: float = 1e-10
    buckets: int = 1 << 18
    backend: str = "stream"
    stream_shards: int = 4
    diagonal_estimator: str = "multinomial"
    diagonal_policy: str = "canonical"
    external_partitions: int = 1
    source_sampler: str = "cdf"

    def __post_init__(self):
        if self.source_sampler not in ("cdf", "multinomial"):
            raise ValueError("source_sampler must be cdf or multinomial")
        if self.backend not in ("stream", "materialized"):
            raise ValueError("backend must be stream or materialized")
        if self.diagonal_estimator not in ("multinomial", "u_statistic"):
            raise ValueError("diagonal_estimator must be multinomial or u_statistic")
        if not 1 <= self.stream_shards <= 32:
            raise ValueError("stream_shards must be 1..32")
        if self.diagonal_policy not in ("canonical", "incremental"):
            raise ValueError("diagonal_policy must be canonical or incremental")
        if self.external_partitions<1 or self.external_partitions&(self.external_partitions-1):
            raise ValueError("external_partitions must be a positive power of two")
        if min(self.source_batch_size, self.max_edges, self.buckets) < 1:
            raise ValueError("Batch, edge limit and buckets must be positive")
        if self.buckets & (self.buckets-1):
            raise ValueError("buckets must be a power of two")
        if not math.isfinite(self.coefficient_threshold) or self.coefficient_threshold < 0:
            raise ValueError("coefficient threshold must be finite and nonnegative")
        if not math.isfinite(self.denominator_epsilon) or self.denominator_epsilon <= 0:
            raise ValueError("denominator epsilon must be finite and positive")


def reduce_complex(keys, values):
    """Diagnostic exact reduction, preserving all words and complex components."""
    unique, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    output = torch.zeros((unique.shape[0], 2), dtype=torch.float64, device=values.device)
    output.index_add_(0, inverse, torch.view_as_real(values))
    return unique, torch.view_as_complex(output)


def source_cdf(probability):
    """Cache a FP64 CDF once; avoid multinomial's 2**24-category limit.

    Caller supplies finite, nonnegative probabilities with positive sum. The
    final entry is normalized to exactly one, so a uniform draw strictly below
    one never falls beyond a trailing zero-probability interval.
    """
    cdf = probability.to(torch.float64).cumsum(0)
    cdf /= cdf[-1].clone()
    return cdf


def sample_source_cdf(cdf, draws, generator):
    uniforms = torch.rand(draws, dtype=torch.float64, device=cdf.device, generator=generator)
    # right=True skips zero-mass intervals, including leading zeros when u=0.
    return torch.searchsorted(cdf, uniforms, right=True)


class StochasticResidentPT2:
    def __init__(self, ham: CudaHamiltonian, states, coefficients, reference_energy,
                 config: StochasticConfig | None = None):
        self.ham = ham
        self.config = config or StochasticConfig()
        self.keys = ham.pack(states).reshape(-1, ham.n_words)
        self.coefficients = torch.as_tensor(coefficients, dtype=torch.complex128, device=ham.device).reshape(-1).contiguous()
        self.energy = torch.as_tensor(reference_energy, dtype=torch.float64, device=ham.device).reshape(())
        if self.keys.shape[0] != self.coefficients.numel() or not self.coefficients.numel():
            raise ValueError("Reference coefficients and nonempty support must agree")
        if not bool(torch.isfinite(self.coefficients).all()) or not bool(torch.isfinite(self.energy)):
            raise ValueError("Reference coefficients/energy must be finite")
        if torch.unique(self.keys, dim=0).shape[0] != self.keys.shape[0]:
            raise ValueError("Reference support must contain unique determinants")
        self.norm = torch.view_as_real(self.coefficients).square().sum()
        self.real_coefficients = not bool(self.coefficients.imag.any())
        if not bool(self.norm > 0):
            raise ValueError("Reference norm must be positive")
        self.probability = self.coefficients.abs()
        self.probability /= self.probability.sum()
        self.source_cdf = source_cdf(self.probability)
        self.index = ham._extension.packed_index(self.keys)
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def denominator(self, keys):
        packed = keys[:, 0] if self.ham.n_words == 1 else keys
        d = self.energy-self.ham._diagonal_packed(packed)
        eps = self.config.denominator_epsilon
        return torch.where(d.abs() < eps, torch.where(d >= 0, eps, -eps), d)

    @torch.no_grad()
    def window(self, draws: int, seed: int, *, mode="sketch", sample_indices=None,
               partition_seed=None, sketch_seed=None):
        if draws < 2:
            raise ValueError("The unbiased quadratic estimator requires at least two draws")
        if mode not in ("sketch", "exact"):
            raise ValueError("mode must be sketch or exact")
        if mode == "exact" and self.world != 1:
            raise ValueError("Exact diagnostic is single-rank; use the distributed sketch mainline")
        cfg, ham = self.config, self.ham
        # All ranks construct identical global draws, then own disjoint source
        # rows. No rank independently squares an incomplete amplitude.
        generator = torch.Generator(device=ham.device).manual_seed(seed)
        with record_function("stochastic/source_sampling"):
            if sample_indices is None:
                sample_indices = (sample_source_cdf(self.source_cdf, draws, generator)
                    if cfg.source_sampler == "cdf" else
                    torch.multinomial(self.probability, draws, replacement=True, generator=generator))
            else:
                sample_indices = torch.as_tensor(sample_indices, dtype=torch.int64, device=ham.device)
                if sample_indices.shape != (draws,) or bool(((sample_indices < 0) | (sample_indices >= len(self.keys))).any()):
                    raise ValueError("Invalid supplied source draws")
                if bool((self.probability[sample_indices] <= 0).any()):
                    raise ValueError("Cannot sample a source with zero probability")
            indices, counts = torch.unique(sample_indices, sorted=True, return_counts=True)
            unique_sources = indices.numel()
            first = indices.numel()*self.rank//self.world
            last = indices.numel()*(self.rank+1)//self.world
            indices, counts = indices[first:last], counts[first:last].to(torch.float64)
            c = self.coefficients[indices]
            probability = self.probability[indices]
            weighted = c*(counts/probability)
            # Treat the within-source diagonal as a linear Monte Carlo sum.
            # For multinomial counts, E[n_i*n_j]=m*(m-1)*p_i*p_j (i!=j)
            # and E[n_i]=m*p_i. This removes redundant count-squared noise
            # without changing the expectation. The U-statistic option uses
            # the count-squared correction for diagnostic comparisons.
            self_factors = (1-(draws-1)*probability/counts if cfg.diagonal_estimator == "multinomial"
                            else counts.reciprocal())
            # This screening target depends on ORIGINAL c, never sampled weights.
            thresholds = cfg.coefficient_threshold/c.abs()
        # Derive separate reproducible streams for source, partition and sketch
        # randomness; explicit seeds allow each component to vary independently.
        if sketch_seed is None:
            sketch_seed = (seed ^ 0x345AD173C9) & ((1 << 63)-1)
        if partition_seed is None:
            partition_seed = (seed ^ 0x6A09E667F3BCC909) & ((1 << 63)-1)
        if not (0 <= sketch_seed < 1 << 63 and 0 <= partition_seed < 1 << 63):
            raise ValueError("Partition and sketch seeds must be nonnegative signed 64-bit integers")
        hash_generator = torch.Generator(device=ham.device).manual_seed(sketch_seed)
        seeds = torch.randint(0, (1 << 63)-1, (1,), dtype=torch.int64, device=ham.device, generator=hash_generator)
        sketch = torch.zeros((1, 2, cfg.buckets, 2), dtype=torch.float64, device=ham.device) if mode == "sketch" else None
        self_term = torch.zeros((), dtype=torch.float64, device=ham.device)
        edge_count = torch.zeros((), dtype=torch.int64, device=ham.device)
        key_chunks, value_chunks = [], []
        for start in range(0, indices.numel(), cfg.source_batch_size):
            stop = min(start+cfg.source_batch_size, indices.numel())
            keys = self.keys[indices[start:stop]].contiguous()
            if mode == "sketch" and cfg.backend == "stream":
                with record_function("stochastic/fused_external_stream"):
                    self_parts, edge_parts = ham._extension.stream_external_words(
                        keys, ham._tables, ham.n_orb, ham.core_energy,
                        self.keys, self.index, thresholds[start:stop].contiguous(),
                        weighted[start:stop].contiguous(), self_factors[start:stop].contiguous(),
                        self.energy, cfg.denominator_epsilon, sketch, seeds, cfg.buckets,
                        self.real_coefficients, cfg.stream_shards, cfg.diagonal_policy == "incremental",
                        cfg.external_partitions, partition_seed)
                    self_term += self_parts.sum()
                    edge_count += edge_parts.sum()
                continue
            packed = keys[:, 0].contiguous() if ham.n_words == 1 else keys
            with record_function("stochastic/coupling"):
                batch = ham._coupled_packed(packed, with_diagonal=cfg.diagonal_policy == "incremental", include_self=False,
                    thresholds=thresholds[start:stop].contiguous(), max_edges=cfg.max_edges)
                source = torch.repeat_interleave(torch.arange(start, stop, device=ham.device),
                                                batch.lengths, output_size=batch.coeffs.numel())
                edge_keys = batch.coupled_states.reshape(-1, ham.n_words)
            with record_function("stochastic/membership"):
                outside = ~ham._extension.packed_contains(self.keys, self.index, edge_keys)
                if cfg.external_partitions>1:
                    outside &= ham._extension.packed_partition(edge_keys, cfg.external_partitions, partition_seed)
                keys = edge_keys[outside].contiguous()
                source = source[outside]
                h = batch.coeffs[outside]
                values = h*weighted[source]
                edge_count += outside.sum()
            with record_function("stochastic/diagonal_and_self"):
                if cfg.diagonal_policy == "incremental":
                    denominator = self.energy-batch.coupled_diag[outside]
                    near = denominator.abs()<1e-4
                    # Empty tensor launches are safe; this is a diagnostic
                    # materialized path, so dynamic boolean indexing is fine.
                    denominator[near] = self.denominator(keys[near].contiguous())
                else:
                    denominator = self.denominator(keys)
                squared = values.real.square()+values.imag.square()
                self_term += (squared*self_factors[source]/denominator).sum()
            if mode == "sketch":
                with record_function("stochastic/sketch_update"):
                    ham._extension.weighted_sketch_words(sketch, keys, values, denominator, seeds, cfg.buckets)
            else:
                key_chunks.append(keys); value_chunks.append(values)
        if self.world > 1:
            with record_function("stochastic/collective"):
                dist.all_reduce(sketch); dist.all_reduce(self_term); dist.all_reduce(edge_count)
        with record_function("stochastic/contract"):
            if mode == "sketch":
                sectors = torch.linalg.vector_norm(sketch.reshape(1, 2, -1), dim=2).square()
                quadratic = sectors[:, 0]-sectors[:, 1]
            elif key_chunks:
                keys, values = reduce_complex(torch.cat(key_chunks), torch.cat(value_chunks))
                quadratic = ((values.real.square()+values.imag.square())/self.denominator(keys)).sum().reshape(1)
            else:
                quadratic = torch.zeros(1, dtype=torch.float64, device=ham.device)
            # An independently refreshed uniform hash partition subsamples
            # external determinants before diagonal/sketch work. Multiplying
            # by P restores the full screened external-space expectation.
            replicas = (quadratic-self_term)*(cfg.external_partitions/(draws*(draws-1)*self.norm))
        return {'estimate':replicas.mean(), 'replica_estimates':replicas, 'self_term':self_term,
                'external_edges':edge_count, 'draws':draws, 'seed':seed,
                'unique_sources':unique_sources, 'partition_seed':partition_seed,
                'sketch_seed':sketch_seed,
                'error_bar_unit':'independent window; one sketch per window',
                'coefficient_threshold':cfg.coefficient_threshold}

    @torch.no_grad()
    def run(self, draws: int, windows: int, seed: int = 0, *, mode="sketch"):
        if windows < 2:
            raise ValueError("At least two independent windows are needed for a total standard error")
        reports = [self.window(draws, seed+104729*k, mode=mode) for k in range(windows)]
        estimates = torch.stack([r['estimate'] for r in reports])
        return {'E2_external':estimates.mean(), 'standard_error':estimates.std(unbiased=True)/math.sqrt(windows),
                'window_estimates':estimates, 'draws_per_window':draws, 'windows':windows,
                'external_edges':torch.stack([r['external_edges'] for r in reports]).sum(),
                'mode':mode, 'uncertainty':'joint source-and-hash Monte Carlo standard error; screening bias excluded'}
