# Changelog

## 0.1.0rc1

- Added distributed semistochastic Epstein–Nesbet perturbation theory on GPUs
  for explicit determinant wavefunctions.
- Included the sorted-list coarse-boundary optimization for paired thresholds.
- Added a public reference format, validated real RHF FCIDUMP/table readers,
  a fixed-setting one-call evaluator, and CLI JSON reports.
- Supported single-GPU and replicated-reference multi-GPU evaluation through the
  same public API, with source tiles distributed across ranks.
- Added a standalone FCIDUMP reader and excitation-table builder with sparse,
  symmetry-compressed spatial integrals and deterministic target ordering.
- Added raw/internal/Olsen projection reporting from caller-supplied `H_SS ψ`.
- Added source-based extension builds, generated examples, CPU/CUDA regression
  tests, optional distributed checks, and release/CI tooling.
