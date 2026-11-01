"""Distributed semistochastic Epstein–Nesbet perturbation theory on GPUs.

Importing the package does not compile extensions or initialize CUDA.
"""
__version__ = "0.1.0rc1"

__all__ = ["__version__", "CudaHamiltonian", "Reference", "PT2Config", "PT2Result",
           "evaluate", "load_fcidump", "load_tables", "save_tables", "internal_correction"]

def __getattr__(name):
    if name == "CudaHamiltonian":
        from ._backend.cuda_coupled import CudaHamiltonian
        return CudaHamiltonian
    if name == "Reference":
        from .reference import Reference
        return Reference
    if name in {"PT2Config", "PT2Result", "evaluate"}:
        from . import evaluator
        return getattr(evaluator, name)
    if name in {"load_fcidump", "load_tables", "save_tables"}:
        from . import integrals
        return getattr(integrals, name)
    if name == "internal_correction":
        from .internal import internal_correction
        return internal_correction
    raise AttributeError(name)
