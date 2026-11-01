"""Portable reference coefficients and determinant bit containers."""
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np


@dataclass(frozen=True)
class Reference:
    """Explicit coefficients in increasing interleaved spin-orbital order.

    keys: uint64/int64 array (N, ceil(n_spin_orbitals/64)), low word first.
    energy: this support's Rayleigh energy with the same Hamiltonian and core.
    Coefficients need not be normalized. Duplicate or zero rows are rejected.
    """
    keys: np.ndarray
    coefficients: np.ndarray
    energy: float
    n_spin_orbitals: int

    def __post_init__(self):
        n = self.n_spin_orbitals
        if type(n) is not int or not 1 <= n <= 512:
            raise ValueError("n_spin_orbitals must be an integer in 1..512")
        keys = np.asarray(self.keys)
        if keys.dtype not in (np.dtype("uint64"), np.dtype("int64")):
            raise ValueError("Packed determinants must use uint64 or int64")
        if keys.ndim == 1 and n <= 64:
            keys = keys[:, None]
        if keys.ndim != 2 or keys.shape[1] != (n+63)//64 or len(keys) == 0:
            raise ValueError("Incorrect packed-key shape or empty support")
        keys = np.array(keys, order="C", copy=True).view(np.uint64)
        if n % 64 and np.any(keys[:, -1] >> np.uint64(n % 64)):
            raise ValueError("Occupied bits outside the Hamiltonian")
        if np.unique(keys, axis=0).shape[0] != len(keys):
            raise ValueError("Duplicate reference determinants")
        coefficients = np.asarray(self.coefficients)
        if coefficients.shape != (len(keys),):
            raise ValueError("Coefficient count does not match support")
        coefficients = np.array(coefficients, dtype=np.complex128, copy=True)
        norm = float(np.vdot(coefficients, coefficients).real)
        if not np.isfinite(coefficients).all() or np.any(coefficients == 0) or not math.isfinite(norm) or norm <= 0:
            raise ValueError("Require finite, nonzero coefficients with finite positive norm")
        if not math.isfinite(float(self.energy)):
            raise ValueError("Nonfinite reference energy")
        keys.flags.writeable = False; coefficients.flags.writeable = False
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "energy", float(self.energy))

    def save(self, path):
        with Path(path).open("wb") as stream:
            np.savez(stream, keys=self.keys, coefficients=self.coefficients,
                     energy=self.energy, n_spin_orbitals=self.n_spin_orbitals)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["keys"], data["coefficients"], float(data["energy"]),
                       int(data["n_spin_orbitals"]))
