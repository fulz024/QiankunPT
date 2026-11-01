"""FP64 CUDA Hamiltonian expansion for packed determinants (1..512 spin orbitals).

Hamiltonian tables are uploaded once, and generated edge arrays remain on the
GPU. ``to_numpy`` explicitly transfers a coupling batch to host arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import torch

from qiankunpt._backend.types import CoupledBatch

_TABLE_NAMES = (
    "single_offsets", "single_targets", "single_values", "double_offsets",
    "double_targets", "double_values", "diagonal_one", "diagonal_two",
)


@lru_cache(maxsize=1)
def _extension():
    from qiankunpt._build import load_extension
    return load_extension("coupled")


@dataclass(frozen=True)
class CudaCoupledBatch:
    ori_states: torch.Tensor
    coupled_states: torch.Tensor
    coeffs: torch.Tensor
    lengths: torch.Tensor
    coupled_diag: torch.Tensor | None

    def to_numpy(self, *, elapsed: float = 0.0) -> CoupledBatch:
        """Synchronize/copy explicitly; retain packed bits including bit 63."""
        return CoupledBatch(
            ori_states=self.ori_states.cpu().numpy().view(np.uint64) if self.ori_states.ndim == 2 else self.ori_states.cpu().numpy(),
            coupled_states=self.coupled_states.cpu().numpy().view(np.uint64) if self.coupled_states.ndim == 2 else self.coupled_states.cpu().numpy(),
            coeffs=self.coeffs.cpu().numpy().astype(np.complex128),
            lengths=self.lengths.cpu().numpy(),
            coupled_diag=None if self.coupled_diag is None else self.coupled_diag.cpu().numpy(),
            py_coupled_elapse=elapsed,
        )


class CudaHamiltonian:
    """Device-side immutable snapshot of a CPU Hamiltonian's excitation tables.

    Reconstruct this object if the Hamiltonian changes. Off-diagonal edges are
    screened by inclusive ``abs(H_ai) >= threshold``. Exact zeros are omitted;
    the self edge is always kept. Rows remain contiguous and the self edge is
    first. Coefficients and diagonals are real FP64.
    """

    def __init__(self, ham, device: torch.device | str = "cuda", *,
                 coupling_layout: str = "warp", diagonal_layout: str = "global"):
        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError("CudaHamiltonian requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.device = torch.device("cuda", torch.cuda.current_device() if dev.index is None else dev.index)
        from qiankunpt.integrals import validate_tables
        tables = validate_tables(ham)
        self.n_orb = int(tables["n_orb"])
        if not 1 <= self.n_orb <= 512:
            raise ValueError("CUDA coupling supports 1..512 spin orbitals")
        self.n_words = (self.n_orb + 63) // 64
        self.core_energy = float(tables["core_energy"])
        self._tables = [torch.from_numpy(tables[name]).to(self.device) for name in _TABLE_NAMES]
        self._extension = _extension()
        if coupling_layout not in ("block", "warp") or diagonal_layout not in ("global", "shared", "indexed"):
            raise ValueError("unknown CUDA kernel layout")
        if self.n_words > 1 and (coupling_layout != "warp" or diagonal_layout == "shared"):
            raise ValueError("Multiword coupling uses warp layout and global/indexed diagonals")
        self.coupling_layout = coupling_layout
        self.diagonal_layout = diagonal_layout

    @property
    def table_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._tables)

    def pack(self, states: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Accept packed words or occupation rows, low orbitals in word zero.

        In occupation rows, values > 0 are occupied.
        A packed Torch tensor uses signed int64 as a bit container, not an index.
        For n>64 packed shape is [rows, ceil(n/64)]; n<=64 retains [rows].
        """
        if isinstance(states, np.ndarray):
            if states.ndim == 1 or (states.ndim == 2 and states.shape[1] == self.n_words and states.dtype in (np.dtype("int64"), np.dtype("uint64"))):
                if states.dtype not in (np.dtype("int64"), np.dtype("uint64")):
                    raise ValueError("Packed determinants must be int64 or uint64")
                states = np.ascontiguousarray(states).view(np.int64)
            states = torch.from_numpy(np.ascontiguousarray(states)).to(self.device)
        elif isinstance(states, torch.Tensor):
            states = states.to(self.device)
        else:
            raise TypeError("states must be a NumPy array or a Torch tensor")
        is_words = states.ndim == 2 and states.shape[1] == self.n_words and states.dtype == torch.int64
        if is_words:
            if states.dtype != torch.int64:
                raise ValueError("Packed words must have int64/uint64 dtype")
            if self.n_words == 1:
                states = states[:, 0]
        elif states.ndim == 2:
            if states.shape[1] != self.n_orb:
                raise ValueError("Occupation width does not match Hamiltonian spin orbitals")
            words = []
            for first in range(0, self.n_orb, 64):
                width = min(64, self.n_orb-first)
                shifts = torch.arange(width, device=self.device, dtype=torch.int64)
                words.append(torch.sum((states[:, first:first+width] > 0).to(torch.int64) << shifts, dim=1))
            states = words[0] if self.n_words == 1 else torch.stack(words, dim=1)
        elif self.n_words != 1 or states.ndim != 1 or states.dtype != torch.int64:
            raise ValueError("Packed determinant shape/dtype does not match Hamiltonian")
        last = states if self.n_words == 1 else states[:, -1]
        remainder = self.n_orb % 64
        if remainder and bool(torch.any((last >> remainder) != 0).item()):
            raise ValueError("Packed determinant contains bits outside the Hamiltonian")
        return states.contiguous()

    def diagonal(self, states: np.ndarray | torch.Tensor) -> torch.Tensor:
        return self._diagonal_packed(self.pack(states))

    def _check_prepared(self, packed: torch.Tensor) -> None:
        # Metadata only: callers must own already validated support keys or
        # outputs of the checked excitation kernel. Public inputs use pack().
        if (not isinstance(packed, torch.Tensor) or packed.device != self.device
                or packed.dtype != torch.int64
                or (packed.ndim != 1 if self.n_words == 1 else packed.ndim != 2 or packed.shape[1] != self.n_words)
                or not packed.is_contiguous()):
            raise ValueError("prepared keys must be contiguous int64 on the Hamiltonian device")

    def _diagonal_packed(self, packed: torch.Tensor) -> torch.Tensor:
        """Private path for owned keys whose bit range was already validated."""
        self._check_prepared(packed)
        if self.n_words > 1 or self.diagonal_layout == "indexed":
            return self._extension.diagonal_words(
                packed.reshape(-1, self.n_words), self._tables[6], self._tables[7],
                self.n_orb, self.core_energy, self.diagonal_layout == "indexed")
        function = (self._extension.diagonal_shared if self.diagonal_layout == "shared"
                    else self._extension.diagonal)
        return function(
            packed, self._tables[6], self._tables[7], self.n_orb, self.core_energy,
        )

    def coupled(self, states: np.ndarray | torch.Tensor, threshold: float = 0.0,
                *, with_diagonal: bool = True, max_edges: int = 20_000_000,
                include_self: bool = True, thresholds: torch.Tensor | None = None) -> CudaCoupledBatch:
        """Expand one source batch on GPU with exact output allocation.

        One scalar device-to-host synchronization obtains the screened edge
        count. No determinant/coupling arrays return to the CPU. ``max_edges``
        fails before output allocation; use ``iter_coupled`` for large supports.
        """
        return self._coupled_packed(self.pack(states), threshold,
                                    with_diagonal=with_diagonal, max_edges=max_edges,
                                    include_self=include_self, thresholds=thresholds)

    def _coupled_packed(self, packed: torch.Tensor, threshold: float = 0.0,
                        *, with_diagonal: bool = True,
                        max_edges: int = 20_000_000, include_self: bool = True,
                        thresholds: torch.Tensor | None = None) -> CudaCoupledBatch:
        """Private expansion of validated support slices; retains edge guard."""
        self._check_prepared(packed)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")
        if max_edges < 0:
            raise ValueError("max_edges must be nonnegative")
        if self.n_words > 1 or not include_self or thresholds is not None:
            return self._coupled_words(packed, threshold, with_diagonal, max_edges, include_self, thresholds)
        count_function = (self._extension.counts_warp if self.coupling_layout == "warp"
                          else self._extension.counts)
        counts = count_function(packed, self._tables, self.n_orb, threshold)
        ends = counts.cumsum(0)
        total = int(ends[-1].item()) if ends.numel() else 0
        if total > max_edges:
            raise MemoryError(f"CUDA expansion needs {total} edges (limit {max_edges}); reduce source_batch_size")
        starts = ends-counts
        lengths = counts.reshape(packed.numel(), 128).sum(1)
        targets = torch.empty(total, dtype=torch.int64, device=self.device)
        values = torch.empty(total, dtype=torch.float64, device=self.device)
        diag = torch.empty(total if with_diagonal else 0, dtype=torch.float64, device=self.device)
        fill_function = (self._extension.fill_warp if self.coupling_layout == "warp"
                         else self._extension.fill)
        fill_function(packed, self._tables, self.n_orb, threshold, self.core_energy,
                      starts, targets, values, diag)
        return CudaCoupledBatch(packed, targets, values, lengths, diag if with_diagonal else None)

    def _coupled_words(self, packed, threshold, with_diagonal, max_edges, include_self, thresholds):
        words = packed.reshape(-1, self.n_words)
        if thresholds is None:
            thresholds = torch.empty(0, dtype=torch.float64, device=self.device)
        else:
            if (thresholds.device != self.device or thresholds.dtype != torch.float64
                    or thresholds.shape != (words.shape[0],) or not thresholds.is_contiguous()):
                raise ValueError("Per-source thresholds must be a contiguous CUDA float64 vector")
            if bool(torch.any(torch.isnan(thresholds) | (thresholds < 0)).item()):
                raise ValueError("Per-source thresholds must be nonnegative and not NaN")
        counts = self._extension.counts_words(words, self._tables, self.n_orb, threshold, thresholds, include_self)
        ends = counts.cumsum(0)
        total = int(ends[-1].item()) if ends.numel() else 0
        if total > max_edges:
            raise MemoryError(f"CUDA expansion needs {total} edges (limit {max_edges}); reduce source_batch_size")
        starts = ends-counts
        lengths = counts.reshape(words.shape[0], 128).sum(1)
        targets = torch.empty((total, self.n_words), dtype=torch.int64, device=self.device)
        values = torch.empty(total, dtype=torch.float64, device=self.device)
        diag = torch.empty(total if with_diagonal else 0, dtype=torch.float64, device=self.device)
        self._extension.fill_words(words, self._tables, self.n_orb, threshold, thresholds,
                                   self.core_energy, include_self, starts, targets, values, diag)
        if self.n_words == 1:
            targets = targets[:, 0]
        return CudaCoupledBatch(packed, targets, values, lengths, diag if with_diagonal else None)

    def iter_coupled(self, states: np.ndarray | torch.Tensor, threshold: float = 0.0,
                     *, source_batch_size: int = 256, with_diagonal: bool = True,
                     max_edges: int = 20_000_000):
        """Yield source-contiguous CUDA batches for downstream streaming."""
        if source_batch_size < 1:
            raise ValueError("source_batch_size must be positive")
        packed = self.pack(states)
        for start in range(0, packed.shape[0], source_batch_size):
            yield self._coupled_packed(packed[start:start+source_batch_size], threshold,
                                       with_diagonal=with_diagonal, max_edges=max_edges)
