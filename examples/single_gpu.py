"""Run after `qiankunpt make-example example-output`."""
from qiankunpt import CudaHamiltonian, PT2Config, Reference, evaluate, load_fcidump

ham = CudaHamiltonian(load_fcidump("example-output/FCIDUMP"), diagonal_layout="indexed")
reference = Reference.load("example-output/reference.npz")
result = evaluate(ham, reference, PT2Config(
    fine_threshold=0.01, coarse_threshold=2.0, target_standard_error=1e-3,
    external_partitions=1, buckets=1 << 18, control_buckets=1 << 18,
))
print(result.to_dict())
assert result.converged
