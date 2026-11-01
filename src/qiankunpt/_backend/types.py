"""Materialized coupling output used by explicit diagnostic calls."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CoupledBatch:
    ori_states: np.ndarray
    coupled_states: np.ndarray
    coeffs: np.ndarray
    lengths: np.ndarray
    py_coupled_elapse: float
    coupled_diag: np.ndarray | None = None
    sampling_metadata: object = None
