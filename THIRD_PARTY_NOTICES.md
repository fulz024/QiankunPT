# Attribution and dependencies

QiankunPT builds on the authors' PT2 implementation and QiankunNet/PNQS
local-energy code, including contributions by 阚博文 (Bowen Kan). The FCIDUMP
reader in `_native/integrals.cpp` is a standalone implementation for this
package. Code and documentation are licensed under BSD-3-Clause.

Runtime/build dependencies are installed separately and retain their own
licenses. In particular, PyTorch, NumPy and Ninja are not relicensed by this
package. PyTorch supplies the pybind11 headers used for extension builds.
NVIDIA CUDA and its development tools must be installed under NVIDIA's terms.

The semistochastic source estimator and deterministic-control construction
follow semistochastic selected-CI methods. CountSketch comes from randomized
linear algebra. Scientific use should acknowledge the associated article and
the method references cited there.
