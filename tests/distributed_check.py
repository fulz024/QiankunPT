"""Opt-in: torchrun --standalone --nproc-per-node=2 tests/distributed_check.py."""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from qiankunpt import CudaHamiltonian, PT2Config, Reference, evaluate


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from qiankunpt.cli import make_example
    from qiankunpt.integrals import load_fcidump
    import tempfile
    # Each rank prepares its own generated input and single-GPU baseline.
    with tempfile.TemporaryDirectory() as temporary:
        files = make_example(temporary)
        ham = CudaHamiltonian(load_fcidump(files["fcidump"]), diagonal_layout="indexed")
        reference = Reference.load(files["reference"])
    config = PT2Config(fine_threshold=.01, coarse_threshold=2.,
        target_standard_error=.001, external_partitions=1,
        buckets=1 << 18, control_buckets=1 << 18,
        safety_gib=.25, memory_gib=2., maximum_windows=8)
    single = evaluate(ham, reference, config)
    np.testing.assert_allclose(single.report["E2_external"], -.5, atol=1e-12)
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        result = evaluate(ham, reference, config)
        assert single.converged and result.converged
        assert result.report["world_size"] == dist.get_world_size()
        np.testing.assert_allclose(result.report["E2_external"], single.report["E2_external"], atol=1e-12)
        if rank == 0:
            print("PASS: public evaluator on one GPU and a replicated-reference NCCL group")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
