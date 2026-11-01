# Contributing

Use a virtual environment and install `python -m pip install -e ".[test]"`.
Run CPU checks before changing a public interface. CUDA changes should include
a small numerical case that exercises the changed path, including phases,
screening boundaries and denominators where relevant. Avoid full-scale
benchmark repetitions that add no information to a code change.

Preserve interleaved spin-orbital order, signed-int64 bit containers, and
complex128 coefficient semantics. Check all words of a determinant key.
Connections belonging to an identical target must add before squaring. Fine
and coarse estimators in a correction window must share source/partition/hash
randomness. Keep the full-source control's sketch uncertainty in the report.

Do not commit generated molecular Hamiltonians, checkpoints, compiled
extensions, machine-specific paths or credentials. Tests should generate their
own small inputs. Keep optional experimental work outside the default API.

For a bug report, include the QiankunPT version, `qiankunpt doctor` output,
compiler/toolkit versions, the smallest shareable generated input, settings,
and traceback. Remove private paths and research data before posting.
