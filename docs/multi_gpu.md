# Distributed semistochastic PT2 on GPUs

QiankunPT distributes source traversal across GPUs and combines the partial
sketch amplitudes with NCCL. Use one process per GPU, with the same reference,
Hamiltonian and evaluation settings on every rank.

## Launch with torchrun

```bash
torchrun --standalone --nproc-per-node=2 -m qiankunpt run \
  --tables tables.npz --reference reference.npz --output result.json
```

The CLI selects `LOCAL_RANK`, initializes NCCL and invokes identical evaluation
calls. Each device holds the full reference, coefficients, membership index and
Hamiltonian tables. Source tiles are assigned across ranks. Partial sketch
arrays and self terms are summed **before** squaring; there is no exchange of
an enumerated external determinant list in this path. Scalars and metadata are
also reduced. The smallest available per-rank memory budget determines the
common settings. Only rank zero writes the report.

For multiple nodes, use the normal `torchrun` node/rendezvous options. An
application must supply identical Hamiltonian tables as well as identical
reference/configuration values on all ranks. The one-call API checks reference
and configuration identity, but does not hash the complete device tables on
every invocation.

All ranks must execute the same collective sequence, including ranks assigned
zero source tiles. There is no automatic recovery from a failed rank. Run distributed
tests on your target topology before relying on new transport configurations.
Per-rank reported CUDA memory excludes NCCL/driver allocations, so allow room
for transport buffers as well as the configured safety reserve.
