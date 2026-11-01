"""Command line interface for distributed semistochastic EN-PT2."""
import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from . import __version__


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def make_example(directory):
    """Generate a two-site Hubbard Hamiltonian and reference determinant."""
    import numpy as np
    from .reference import Reference
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    integral = directory / "FCIDUMP"
    ref = directory / "reference.npz"
    if integral.exists() or ref.exists():
        raise FileExistsError("Example inputs already exist; choose an empty output directory")
    integral.write_text(" &FCI NORB=2,NELEC=2,MS2=0,\n ORBSYM=1,1,\n ISYM=1,\n &END\n"
        " 4.0 1 1 1 1\n 4.0 2 2 2 2\n -1.0 2 1 0 0\n 0.0 0 0 0 0\n", encoding="utf-8")
    # |alpha_0 beta_1>: E_S=0; two external hopping targets at diagonal U=4,
    # so the unscreened external EN-PT2 correction is exactly -2*t**2/U=-0.5.
    Reference(np.array([[9]], dtype=np.uint64), np.array([1.]), 0., 4).save(ref)
    return dict(fcidump=str(integral), reference=str(ref), exact_E2_external=-0.5)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="qiankunpt", description="Distributed semistochastic Epstein–Nesbet perturbation theory on GPUs")
    parser.add_argument("--version", action="version", version=f"QiankunPT {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Inspect runtime; optionally compile the bundled extensions")
    doctor.add_argument("--build", action="store_true")
    example = sub.add_parser("make-example", help="Write a small generated Hubbard input and reference")
    example.add_argument("directory", type=Path)
    export = sub.add_parser("export-tables", help="Build and validate excitation tables once on the CPU")
    export.add_argument("fcidump", type=Path); export.add_argument("output", type=Path)
    run = sub.add_parser("run", help="Run semistochastic EN-PT2 on one GPU or a torchrun NCCL group")
    inputs = run.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--fcidump", type=Path); inputs.add_argument("--tables", type=Path)
    run.add_argument("--reference", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--device", default="cuda")
    run.add_argument("--config", type=Path, help="JSON PT2Config fields; CLI options override them")
    for flag in ("fine-threshold", "coarse-threshold", "target-standard-error", "maximum-seconds", "memory-gib", "safety-gib"):
        run.add_argument("--"+flag, type=float)
    for flag in ("seed", "minimum-windows", "maximum-windows", "unique-sources", "external-partitions", "buckets", "control-buckets"):
        run.add_argument("--"+flag, type=int)
    args = parser.parse_args(argv)
    if args.command == "make-example":
        print(json.dumps(make_example(args.directory), indent=2)); return 0
    if args.command == "doctor":
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        info = dict(package="QiankunPT", version=__version__, python=sys.version.split()[0],
            torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_toolkit=CUDA_HOME,
            cuda_available=torch.cuda.is_available(), devices=torch.cuda.device_count())
        if args.build:
            from ._build import load_extension
            info["integral_extension"] = load_extension("integrals").__name__
            if torch.cuda.is_available():
                info["cuda_extension"] = load_extension("coupled").__name__
        print(json.dumps(info, indent=2)); return 0
    if args.command == "export-tables":
        from .integrals import load_fcidump, save_tables
        if args.output.exists():
            parser.error("Output already exists")
        save_tables(args.output, load_fcidump(args.fcidump)); return 0
    import torch
    import torch.distributed as dist
    from . import CudaHamiltonian, PT2Config, Reference, evaluate, load_fcidump, load_tables
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    start = time.perf_counter()
    metadata = {}
    try:
        if args.output.exists():
            raise FileExistsError("Output already exists; choose a new result path")
        overrides = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        for field in fields(PT2Config):
            value = getattr(args, field.name, None)
            if value is not None:
                overrides[field.name] = value
        config = PT2Config(**overrides)
        source = args.tables or args.fcidump
        metadata = dict(input_sha256={"tables" if args.tables else "fcidump": _sha256(source),
                                      "reference": _sha256(args.reference)})
        tables = load_tables(source) if args.tables else load_fcidump(source)
        reference = Reference.load(args.reference)
        ham = CudaHamiltonian(tables, device="cuda" if distributed else args.device, diagonal_layout="indexed")
        torch.cuda.synchronize(ham.device)
        metadata["initialization_seconds"] = time.perf_counter()-start
        def progress(report):
            _save(args.output, dict(report, **metadata))
        result = evaluate(ham, reference, config, progress=progress)
        if rank == 0:
            report = dict(result.to_dict(), **metadata, process_seconds=time.perf_counter()-start)
            _save(args.output, report)
            print(json.dumps({k: report.get(k) for k in ("status", "E2_external", "total_energy", "standard_error", "pt_seconds")}, indent=2))
        return 0 if result.converged else 2
    except Exception as error:
        if rank == 0:
            # Preserve completed-window evidence if a later stage fails.
            previous = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
            # Never replace a pre-existing unrelated output on an input error.
            if not isinstance(error, FileExistsError):
                _save(args.output, dict(previous, **metadata, status="FAILED", error=str(error)))
        raise
    finally:
        if distributed:
            dist.destroy_process_group()
