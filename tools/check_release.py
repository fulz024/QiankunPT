"""Check source contents, release file types and accidental local paths/secrets."""
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {"README.md", "LICENSE", "pyproject.toml", "MANIFEST.in", "CITATION.cff",
    "CHANGELOG.md", "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md", ".gitignore", ".gitattributes"}
FOLDERS = {"src", "tests", "docs", "examples", "tools", ".github"}
SUFFIXES = {".py", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".md", ".yml", ".typed"}
GENERATED_FOLDERS = {".git", ".venv", "build", "dist", ".pytest_cache"}


def public_files():
    files = []
    for directory, folders, names in os.walk(ROOT):
        parent = Path(directory)
        folders[:] = sorted(folder for folder in folders
                            if folder != "__pycache__" and not folder.endswith(".egg-info")
                            and not (parent == ROOT and folder in GENERATED_FOLDERS))
        for name in [*folders, *sorted(names)]:
            path = parent/name
            relative = path.relative_to(ROOT)
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                raise SystemExit(f"Symlink/outside source path: {relative}")
            if path.is_dir():
                if parent == ROOT and name not in FOLDERS:
                    raise SystemExit(f"Unexpected source directory: {relative}")
                continue
            if parent == ROOT and name == "SOURCE_MANIFEST.json":
                continue
            if relative.as_posix() in ROOT_FILES or (relative.parts[0] in FOLDERS and path.suffix in SUFFIXES):
                files.append(path)
            else:
                raise SystemExit(f"Unexpected source file: {relative}")
    yield from sorted(files)


def check():
    files = list(public_files())
    forbidden = [r"/public/home/", r"/home/[^\s/]+/", r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
                 r"gh[pousr]_[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9]{32,}"]
    failures = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.stat().st_size > 4 << 20:
            failures.append(relative+": oversized source file")
        content = path.read_text(encoding="utf-8-sig")
        # This checker contains the literal detection patterns, not credentials.
        if relative == "tools/check_release.py":
            continue
        if any(re.search(pattern, content) for pattern in forbidden):
            failures.append(relative+": machine path or credential-like text")
    for required in ROOT_FILES:
        if not (ROOT/required).exists():
            failures.append("Missing "+required)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {len(files)} public software files; no detected credentials/machine paths")
    return files


if __name__ == "__main__":
    check()
