"""Inspect distribution members without extracting or executing them."""
import argparse
from pathlib import Path
import tarfile
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=root/"dist")
    args = parser.parse_args()
    artifacts = sorted(p for p in args.directory.iterdir() if p.name.endswith((".whl", ".tar.gz", ".zip")))
    if not artifacts:
        raise SystemExit("Build distribution files first")
    required = ["qiankunpt/_backend/cuda/screening.cuh", "qiankunpt/_backend/cuda/multiword_kernel.cu",
        "qiankunpt/_native/integrals.cpp"]
    for artifact in artifacts:
        if artifact.name.endswith(".tar.gz"):
            with tarfile.open(artifact) as stream:
                names = stream.getnames()
        else:
            with zipfile.ZipFile(artifact) as stream:
                names = stream.namelist()
        for name in names:
            parts = Path(name).parts
            if ".." in parts or name.startswith(("/", "\\")):
                raise SystemExit("Unsafe archive member: "+name)
            if any(p in {".git", ".venv", "__pycache__", ".pytest_cache", "build",
                          "output", "results", "checkpoints"} or p.startswith(".test-") for p in parts):
                raise SystemExit("Generated/cache archive member: "+name)
            if name.endswith((".so", ".pyd", ".dll", ".obj", ".o", ".pyc", ".pt", ".pth",
                              ".ckpt", ".npz", ".npy", ".log", ".ipynb")):
                raise SystemExit("Unexpected binary/data member: "+name)
        for needed in required:
            if not any(n.endswith(needed) for n in names):
                raise SystemExit(f"{artifact.name} missing {needed}")
        print(f"PASS: {artifact.name}, {len(names)} members, required CUDA/native source and headers present")


if __name__ == "__main__":
    main()
