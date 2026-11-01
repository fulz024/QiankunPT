# QiankunPT

**Distributed semistochastic Epstein–Nesbet perturbation theory on GPUs.**

QiankunPT evaluates the external Epstein–Nesbet PT2 correction from reference
determinants, their real or complex coefficients, and a real molecular
Hamiltonian. It combines a deterministic control with source sampling and a
single CountSketch per independent window. Couplings, reference-membership
tests, target diagonals, and sketch updates run on GPUs. In distributed runs,
source tiles are divided across ranks and sketch amplitudes are combined with
NCCL before contraction.

Version **0.1.0rc1** provides a standalone evaluator for wavefunctions supplied
by selected-CI solvers, neural quantum states, or other determinant methods.

## Install

Use Python 3.10 or newer, a CUDA-enabled PyTorch installation, an NVIDIA GPU,
and a CUDA development toolkit compatible with that PyTorch installation.
The toolkit must include `nvcc`. A C++17 compiler and Ninja are also required.
Linux is the main multi-GPU platform;
Windows single-GPU development requires an MSVC developer shell.

After installing the appropriate PyTorch build in your environment:

```bash
cd QiankunPT
python -m pip install -e ".[test]"
qiankunpt doctor
qiankunpt doctor --build
```

The bundled CPU reader and CUDA extensions compile on first use and are cached
by source identity. Set `MAX_JOBS=2` to limit compiler memory,
`TORCH_CUDA_ARCH_LIST` to the deployment architecture, and `TORCH_EXTENSIONS_DIR`
to a writable cache if the default cache is unsuitable.
See [installation](docs/installation.md).

## Run the small example

```bash
qiankunpt make-example example-output
qiankunpt run --fcidump example-output/FCIDUMP \
  --reference example-output/reference.npz --output example-output/result.json \
  --fine-threshold 0.01 --coarse-threshold 2.0 \
  --external-partitions 1 --target-standard-error 0.001 \
  --buckets 262144 --control-buckets 262144
```

The generated two-site Hubbard example has external PT2 correction **−0.5 Ha**.
To reuse prepared excitation tables, run
`qiankunpt export-tables FCIDUMP tables.npz` and replace `--fcidump` with `--tables`.

```python
from qiankunpt import CudaHamiltonian, PT2Config, Reference, evaluate, load_fcidump

ham = CudaHamiltonian(load_fcidump("FCIDUMP"), diagonal_layout="indexed")
reference = Reference.load("reference.npz")
result = evaluate(ham, reference, PT2Config(
    fine_threshold=5e-8, coarse_threshold=5e-7,
    target_standard_error=1e-5,
))
print(result.to_dict())
```

Supply the reference coefficients and their finite-support Rayleigh energy
using the [input conventions](docs/inputs.md).

## What is included

- FP64 coupling and diagonal kernels for up to 512 spin orbitals, with full-key
  determinant membership and real or complex128 reference coefficients.
- A fixed source budget, analytic expected-occupancy calculation, and
  memory-based bucket sizing, with parameters fixed before evaluation.
- Fine and coarse thresholds evaluated in one traversal. Magnitude-sorted
  double-excitation lists stop coarse updates after crossing the coarse cutoff;
  occupied-dependent single excitations retain their individual cutoff tests.
- A one-call semistochastic PT2 API for single-GPU evaluation and distributed
  execution with [`torchrun` and NCCL](docs/multi_gpu.md).
- An optional projected internal correction from a caller-supplied `H_SS ψ`.
  Raw internal, Olsen, and projected terms are reported separately.
- A standard real RHF FCIDUMP reader, portable NumPy reference/table files,
  JSON reports, generated examples, tests, and GitHub Actions workflows.

Hamiltonian tables and the full reference reside on each device. Memory use
depends on the integral sparsity and reference size as well as the orbital count.

## Results and uncertainty

The default source budget is `U=262144`, with `K=floor(U/4)` largest-probability
sources included each window and the draw count determined from expected
distinct sources in the remaining distribution. The default external partition
count is 16. Bucket counts follow available device memory. Settings are frozen
before production, and at least six sampled windows are evaluated.

`TARGET_REACHED` means that the **reported statistical SE** met the requested
target. It does not establish total accuracy. Threshold bias, denominator
regularization, reference quality, rare-event coverage, and stopping effects
need separate assessment. See [method and uncertainty](docs/method.md).

## Associated paper data

Numerical tables and plotting scripts for the accompanying manuscript are
available in the [GPU PT2 paper data repository](https://github.com/fulz024/2026-gpu-pt2-data-deposit).

## Tests and contributing

```bash
python -m pytest -m "not cuda and not distributed"
python -m pytest -m cuda
python tools/make_release.py
```

CPU checks include an actual build of the reader. CUDA tests require a suitable
local GPU/toolkit. Multi-GPU tests are opt-in; see [testing](docs/testing.md).
See [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md), and
[release checklist](docs/releasing.md).

## License and attribution

Code and documentation use the [BSD 3-Clause License](LICENSE).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [CITATION.cff](CITATION.cff).
