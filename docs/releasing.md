# Release checklist

1. Check version consistency in `pyproject.toml`, `src/qiankunpt/__init__.py`,
   `CITATION.cff` and `CHANGELOG.md`. Review the license and attribution.
2. Run the CPU, CUDA and distributed checks described in [testing](testing.md)
   on the supported environments.
3. Install build tools and check the source tree:

   ```bash
   python -m pip install -e ".[test,release]"
   python tools/check_release.py
   ```

4. Build and inspect the distributions:

   ```bash
   python -m build
   python -m twine check dist/*
   python tools/make_release.py
   python tools/check_artifacts.py
   ```

   `make_release.py` writes a source ZIP, a per-file SHA256 manifest and artifact
   checksums. Use `--output PATH` to select another destination.
5. Extract the source ZIP into a clean directory and verify each file against
   `SOURCE_MANIFEST.json`. Run the source checks and software tests there.
6. Install the wheel in a clean environment outside the checkout. Check
   `qiankunpt --version`, run `qiankunpt doctor --build`, and evaluate the
   generated Hubbard example.
7. Publish the distributions, checksums and release notes together.
