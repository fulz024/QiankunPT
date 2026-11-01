"""Validated, real molecular Hamiltonian tables in interleaved spin order."""
from functools import lru_cache
from pathlib import Path
import math
import re
import numpy as np

TABLE_NAMES = ("single_offsets", "single_targets", "single_values", "double_offsets",
               "double_targets", "double_values", "diagonal_one", "diagonal_two")


def validate_tables(tables):
    """Validate layout before native kernels read offsets or sorted lists."""
    if not isinstance(tables, dict):
        raise TypeError("Expected a table dictionary; use load_fcidump() or load_tables()")
    n = int(tables["n_orb"])
    if n != tables["n_orb"] or not 1 <= n <= 512:
        raise ValueError("Require 1..512 spin orbitals")
    core = float(tables["core_energy"])
    if not math.isfinite(core):
        raise ValueError("Nonfinite core energy")
    result = dict(n_orb=n, core_energy=core)
    for name in TABLE_NAMES:
        value = np.asarray(tables[name])
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite flat array")
        integer = name.endswith(("offsets", "targets"))
        if integer and value.dtype.kind not in "iu":
            raise ValueError(f"{name} requires integer indices")
        dtype = np.int64 if name.endswith("offsets") else np.int32 if integer else np.float64
        if name.endswith("targets") and ((value < 0).any() or (value >= n).any()):
            raise ValueError(f"{name} contains an out-of-range orbital")
        if name.endswith("offsets") and value.dtype.kind == "u" and (value > np.iinfo(np.int64).max).any():
            raise ValueError("Offset exceeds signed 64-bit range")
        result[name] = np.ascontiguousarray(value, dtype=dtype)
    for prefix, rows, entries in (("single", n, len(result["single_targets"])),
                                  ("double", n*n, len(result["double_values"]))):
        offsets = result[prefix + "_offsets"]
        if len(offsets) != rows+1 or offsets[0] != 0 or offsets[-1] != entries or (np.diff(offsets) < 0).any():
            raise ValueError(f"Invalid {prefix} offsets")
    if len(result["single_values"]) != len(result["single_targets"]) * (n+1):
        raise ValueError("Single entries require a constant and n spectator coefficients")
    if len(result["double_targets"]) != 2*len(result["double_values"]):
        raise ValueError("Each double entry requires two targets")
    magnitude = np.abs(result["double_values"])
    rises = np.flatnonzero(magnitude[1:] > magnitude[:-1]) + 1
    if not np.isin(rises, result["double_offsets"]).all():
        raise ValueError("Double excitation lists must be sorted by descending magnitude")
    if len(result["diagonal_one"]) != n or len(result["diagonal_two"]) != n*n:
        raise ValueError("Incorrect diagonal table sizes")
    pairs = result["diagonal_two"].reshape(n, n)
    if not np.array_equal(pairs, pairs.T) or np.any(np.diag(pairs)):
        raise ValueError("Pair diagonal table must be symmetric with a zero diagonal")
    for key in ("n_elec", "ms2"):
        if key in tables:
            result[key] = int(tables[key])
    return result


@lru_cache(maxsize=1)
def _reader():
    from ._build import load_extension
    return load_extension("integrals")


def load_fcidump(path):
    """Read a standard real RHF spatial-orbital FCIDUMP.

    NORB counts spatial orbitals here. The returned n_orb counts spin orbitals.
    E_core must be present and include any folded frozen-core contribution.
    UHF/spinor formats and complex integrals are not supported by this reader.
    """
    path = Path(path).resolve(strict=True)
    with path.open(encoding="utf-8") as stream:
        lines = []
        for line in stream:
            lines.append(line)
            if "&END" in line or "/" in line:
                break
        else:
            raise ValueError("Missing FCIDUMP header terminator")
        header = " ".join(lines)
        fields = {}
        for name in ("NORB", "NELEC", "MS2"):
            match = re.search(r"\b" + name + r"\s*=\s*(-?\d+)", header)
            if not match:
                raise ValueError(f"Missing uppercase {name} in FCIDUMP header")
            fields[name] = int(match[1])
        n, ne, ms = fields["NORB"], fields["NELEC"], fields["MS2"]
        if not 1 <= n <= 256 or not 0 <= ms <= ne <= 2*n or (ne-ms) % 2:
            raise ValueError("Require NORB<=256 and a valid nonnegative MS2 electron sector")
        uhf = re.search(r"\bUHF\s*=\s*([^,\s]+)", header)
        if uhf and uhf[1].strip(".").upper() != "FALSE":
            raise ValueError("Only real RHF spatial-orbital FCIDUMP is supported")
        found_core = False
        # Validate before compiling or entering the table builder. This pass
        # is streaming and does not retain the integral file in Python memory.
        for line in stream:
            if not line.strip():
                continue
            columns = line.split()
            if len(columns) != 5:
                raise ValueError("FCIDUMP records must contain value i j k l")
            try:
                value = float(columns[0]); i, j, k, l = map(int, columns[1:])
            except ValueError as error:
                raise ValueError("Use real integral records with E-format exponents") from error
            if not math.isfinite(value) or not all(0 <= x <= n for x in (i,j,k,l)):
                raise ValueError("Nonfinite integral or orbital index outside NORB")
            if not ((i and j and k and l) or (i and j and k == l == 0)
                    or (i and j == k == l == 0) or i == j == k == l == 0):
                raise ValueError("Malformed FCIDUMP integral indices")
            found_core |= i == j == k == l == 0
        if not found_core:
            raise ValueError("FCIDUMP must include an explicit 0 0 0 0 core-energy record")
    return validate_tables(dict(_reader().load_tables(str(path), n, ne, ms)))


def save_tables(path, tables):
    """Save validated tables without pickle or executable Python objects."""
    with Path(path).open("wb") as stream:
        np.savez(stream, **validate_tables(tables))


def load_tables(path):
    """Load a table archive produced by save_tables()."""
    with np.load(path, allow_pickle=False) as archive:
        return validate_tables({name: archive[name] for name in archive.files})
