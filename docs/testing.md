# Testing and validation scope

```bash
python -m pytest -m "not cuda and not distributed"
python -m pytest -m cuda
torchrun --standalone --nproc-per-node=2 tests/distributed_check.py
```

The first command builds the CPU reader and checks input validation, full-word
round trips, expected source occupancy against enumerated draws, memory sizing,
window variance arithmetic, and a complex projected internal correction against
an independent constrained linear solve. Use `-m "not native and not cuda and
not distributed"` for Python-only checks without a compiler.

Reader checks compare a generated multi-electron Hamiltonian against a dense
fermion-operator construction and verify integral-permutation symmetry and
duplicate-record invariance.

CUDA checks compare a small Hubbard Hamiltonian action to independently applied
fermionic operators, exercise the complete public PT2 call, and check paired
fine/coarse threshold fields against materialized contraction. Boundary cases
include equality, a value immediately below the cutoff, zero matrix elements,
positive/negative/floored denominators, complex coefficients, high words and
occupied bit 63.

The separate distributed command compares the public evaluator's single-GPU
result with its replicated-reference NCCL result. The one-determinant example
also exercises ranks with no assigned source tiles. It needs two NVIDIA
GPUs, Linux, NCCL and a compatible CUDA development environment. It is not run
on GitHub's ordinary CPU-hosted runners. The manual GPU workflow is intended
for a trusted self-hosted runner and does not execute untrusted pull requests.

The tests check numerical identities and execution paths. Statistical SE
coverage in rare-event or large molecular calculations requires additional
validation, as do the physical model, reference quality and spin state of a
particular calculation.
