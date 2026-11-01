# Method and statistical report

QiankunPT implements distributed semistochastic Epstein–Nesbet perturbation
theory on GPUs. For a finite reference support S, the external second-order
correction is

```
E2 = sum(a outside S) |sum(i in S) H_ai c_i|^2 / (E_S - H_aa) / sum(i) |c_i|^2.
```

Screening retains nonzero connections with `abs(H_ai*c_i) >= epsilon`.
The original coefficient sets this test, including when sampled coefficients
are reweighted. Denominators with magnitude below `denominator_epsilon` are
replaced by that signed floor; zero receives the positive floor. This defines
a regularized screened calculation and must be disclosed when it matters.

## Deterministic control and sampled correction

The full source support is visited at a coarse threshold for the
**deterministic control**. The difference between fine and coarse thresholds
is evaluated using the same sampled sources, target partition and hash map
within each window. The largest-probability K sources are always included;
the complement is drawn with replacement from probabilities proportional to
`abs(c_i)`. Multinomial self terms correct the quadratic sampling estimator.

Default `U=262144` limits the expected distinct source workload, `K=floor(U/4)`
is capped to leave a nonempty stochastic complement, and the complement draw
count m solves `sum_i [1-(1-p_i)^m] >= U-K`, subject to workspace limits.
The report includes the attainable expected unique count and any cap hit.
This is a fixed preset, not an assertion of optimal U/K/P for every molecule.

An independently refreshed uniform target hash retains one of P external
partitions and scales the estimate by P. It reduces retained target work;
generating and screening candidate connections still costs time. Default P=16.

## CountSketch and GPU execution

Each retained target contributes its weighted complex amplitude to a shared
bucket with a key-dependent phase from `{1,-1,i,-i}`. Contributions to an
identical determinant add before squaring. Different-key cross terms cancel
in expectation under the sketch model. Positive and negative denominator
sectors are accumulated separately and subtracted after contraction.

There is **one sketch per independent window**. Source, partition and sketch
randomness refresh between windows. The deterministic control visits all
sources but its sketch contraction remains randomized. Its noise cannot be
ignored or assigned zero SE.

The fused CUDA producer generates at the fine threshold once. Sorted double
lists stop at the fine cutoff and stop coarse updates at the coarse cutoff.
Singles need individual tests because their actual matrix elements depend on
the occupied orbitals. Reference targets are excluded using a full-key index;
target diagonals and bucket updates are fused into retained-connection work.

In distributed evaluation, each GPU holds the reference and Hamiltonian tables
and processes a distinct set of source tiles. NCCL reductions combine sketch
amplitudes and self terms before the quadratic contraction. This preserves
interference between contributions from different ranks. See
[distributed execution](multi_gpu.md) for launch and communication requirements.

## Memory and stopping

Bucket counts are powers of two selected from live memory, after reference,
tables, source workspace and a safety reserve. The correction stage budgets
160 bytes/bucket for four fields and one contraction temporary; the control
budgets 32 bytes/bucket. OOM remains possible through allocator/external memory
pressure; these estimates are not an allocator guarantee. Settings do not
change during production.

The one-call evaluator uses a control norm bound derived under independent
ideal complex CountSketch maps, plus the empirical variance of independent
sampled windows. It permits at most 16 control passes and requires at least
six sampled windows by default. Reaching the requested **reported statistical
SE** is the stopping condition. Exceeding a time/window/control limit returns
an explicit incomplete status; the CLI then exits with code 2 and keeps the
partial report. A time limit is checked between windows and does not interrupt
an in-flight CUDA kernel.

The report is not a total-error certificate or an anytime confidence interval.
Finite-window variance can miss rare important determinants. Screening bias,
regularization, imperfect reference states and scientific model choices are
outside the statistical SE. Calibrating coverage requires independent evidence.
The explicit lower-level `engine.run()` instead combines sample variances from
multiple control and correction windows; its error accounting is documented
separately in that method and does not silently use the one-call control bound.

## Optional internal correction

`internal_correction(c, H_SS_c, diagonal)` evaluates the projected diagonal
correction on the supplied support. It reports raw internal PT2, Olsen, and
their projected sum. The projected term already contains Olsen; do not add it
twice. Add only the projected internal term to `E_S + E2_external` for the full
correction. The finite-support residual can be nonzero for a variational neural
wavefunction; no selected-CI convergence condition is imposed by this helper.
