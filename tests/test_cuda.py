from itertools import combinations
import numpy as np
import pytest
import torch
from qiankunpt import CudaHamiltonian, PT2Config, Reference, evaluate, load_fcidump
from qiankunpt.cli import make_example

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


def _independent_hubbard_action(keys):
    """Fermion-operator oracle, independent of excitation-table generation."""
    index = {key: i for i, key in enumerate(keys)}
    matrix = np.zeros((len(keys), len(keys)))
    for col, key in enumerate(keys):
        matrix[col,col] = 4*((key & 3) == 3) + 4*((key & 12) == 12)
        for p,q in ((0,2),(2,0),(1,3),(3,1)):
            if not key & (1 << q) or key & (1 << p):
                continue
            phase = (-1)**((key & ((1 << q)-1)).bit_count())
            target = key ^ (1 << q)
            phase *= (-1)**((target & ((1 << p)-1)).bit_count())
            target |= 1 << p
            matrix[index[target], col] += -phase
    return matrix


def test_coupling_diagonal_and_complete_public_evaluation(tmp_path):
    example = make_example(tmp_path/"demo")
    ham = CudaHamiltonian(load_fcidump(example["fcidump"]), diagonal_layout="indexed")
    keys = [sum(1 << i for i in occ) for occ in combinations(range(4),2)]
    expected = _independent_hubbard_action(keys)
    out = ham.coupled(np.array(keys, dtype=np.int64)).to_numpy()
    actual = np.zeros_like(expected); cursor = 0
    for col, length in enumerate(out.lengths):
        for key, value in zip(out.coupled_states[cursor:cursor+length], out.coeffs[cursor:cursor+length]):
            actual[keys.index(int(key)), col] += value.real
        cursor += length
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)
    np.testing.assert_allclose(ham.diagonal(np.array(keys, dtype=np.int64)).cpu(), np.diag(expected), rtol=0, atol=1e-13)
    reference = Reference.load(example["reference"])
    result = evaluate(ham, reference, PT2Config(fine_threshold=.01, coarse_threshold=2.,
        target_standard_error=.001, external_partitions=1, buckets=1 << 18, control_buckets=1 << 18,
        safety_gib=.25, memory_gib=2., maximum_windows=8))
    assert result.converged
    report = result.to_dict()
    np.testing.assert_allclose(report["E2_external"], -.5, atol=1e-12)
    assert len(report["tail"]) == 6 and report["frozen_settings"]["deterministic_sources"] == 0
