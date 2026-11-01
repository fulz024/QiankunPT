"""Build from bundled source with a content-specific PyTorch cache identity."""
import hashlib
import os
from pathlib import Path


def load_extension(kind):
    from torch.utils.cpp_extension import load
    root = Path(__file__).parent
    if kind == "coupled":
        folder = root / "_backend/cuda"
        sources = [folder / n for n in ("coupled_bindings.cpp", "coupled_kernel.cu",
                   "resident_kernels.cu", "multiword_kernel.cu", "packed_ops.cu")]
    elif kind == "integrals":
        folder = root / "_native"
        sources = [folder / "integrals.cpp"]
    else:
        raise ValueError("Unknown extension")
    # Header changes also invalidate the extension. PyTorch additionally tracks
    # compiler arguments and its own ABI in the cache/build directory.
    identity = hashlib.sha256()
    for path in sorted({*sources, *folder.glob("*.h"), *folder.glob("*.hpp"), *folder.glob("*.cuh")}):
        identity.update(path.name.encode()); identity.update(path.read_bytes())
    options = dict(name=f"qiankunpt_{kind}_{identity.hexdigest()[:16]}",
        sources=[str(p) for p in sources],
        extra_cflags=["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"],
        verbose=os.environ.get("QIANKUNPT_VERBOSE", "0") == "1")
    if kind != "integrals":
        options["extra_cuda_cflags"] = ["-O3", "--fmad=false", "-std=c++17"]
    return load(**options)
