# Installation and builds

## Requirements

- Python >=3.10, NumPy >=1.23, PyTorch >=2.3, and Ninja.
- For PT2 evaluation: an NVIDIA GPU, a CUDA-enabled PyTorch build, a compatible
  CUDA development toolkit, and a C++17 compiler. Input reading and
  internal-correction checks can also run on the CPU.
- For multi-GPU evaluation: Linux and a PyTorch build with NCCL. Select one
  process per GPU. Configure inter-node transport through the normal PyTorch
  distributed environment when using more than one node.

Select a CUDA-enabled PyTorch build for your machine and install it before
QiankunPT. The CUDA development toolkit must include `nvcc`.

```bash
python -m pip install -e ".[test]"
qiankunpt doctor
```

`doctor` reports the Python/PyTorch versions, CUDA toolkit location and available
devices. `doctor --build` builds the CPU reader and, when CUDA is available, the
coupling extension.

## First-use compilation

The wheel contains C++/CUDA **source** compiled for the installed PyTorch and
CUDA environment using `torch.utils.cpp_extension`. Cache names include
the source and header content. Retain the PyTorch/compiler combination when
reusing a cache; rebuild after changing environments.

Useful environment variables:

| Variable | Purpose |
|---|---|
| `CUDA_HOME` | CUDA development toolkit location when automatic discovery fails |
| `TORCH_CUDA_ARCH_LIST` | GPU architectures to compile for |
| `TORCH_EXTENSIONS_DIR` | Writable cache outside a small home directory |
| `MAX_JOBS` | Compiler parallelism; use 1 or 2 on memory-constrained hosts |
| `QIANKUNPT_VERBOSE=1` | Show extension build output |

On Windows, run in an x64 MSVC developer shell, set `PYTHONUTF8=1`, and make
`cl`, `nvcc` and `ninja` available. Multi-GPU NCCL execution requires Linux.
Compilation can take minutes and is separate from reported PT2 evaluation time.
See [testing](testing.md) for CPU, CUDA and distributed checks on your environment.
