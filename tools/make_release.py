"""Create a reproducible source archive with SHA256 manifests."""
import ast
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from check_release import ROOT, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"dist", help="Destination for local release artifacts")
    args = parser.parse_args()
    files = check()
    module = ast.parse((ROOT/"src/qiankunpt/__init__.py").read_text(encoding="utf-8"))
    version = next(ast.literal_eval(node.value) for node in module.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "__version__"
                           for target in node.targets))
    prefix = f"QiankunPT-{version}"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    archive = output/(prefix+".zip")
    # ZIP's earliest timestamp makes archives independent of source mtimes.
    timestamp = (1980, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in files:
            item = zipfile.ZipInfo(prefix+"/"+path.relative_to(ROOT).as_posix(), timestamp)
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o644 << 16
            stream.writestr(item, path.read_bytes())
        item = zipfile.ZipInfo(prefix+"/SOURCE_MANIFEST.json", timestamp)
        item.compress_type = zipfile.ZIP_DEFLATED
        stream.writestr(item, json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    (output/"SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    artifacts = [archive, *sorted(output.glob("qiankunpt-"+version+"*.whl")),
                 *sorted(output.glob("qiankunpt-"+version+".tar.gz"))]
    checksums = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in artifacts)
    (output/"SHA256SUMS").write_text(checksums, encoding="utf-8")
    print(f"Created {archive.name}: {archive.stat().st_size} bytes; SHA256 {digest}")


if __name__ == "__main__":
    main()
