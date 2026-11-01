import itertools
import math
import numpy as np
import pytest
import torch
from qiankunpt import PT2Config
from qiankunpt.policy import choose_fixed_sources, invert_expected_unique, single_sketch_bucket_budget
from qiankunpt.statistics import control_variance_bound, estimate


def test_analytic_occupancy_matches_independent_enumeration():
    p = np.array([.1, .2, .7])
    for m in range(2, 6):
        enumerated = sum(len(set(ids))*np.prod(p[list(ids)])
            for ids in itertools.product(range(3), repeat=m))
        analytic = np.sum(-np.expm1(m*np.log1p(-p)))
        np.testing.assert_allclose(enumerated, analytic)
    expectation = lambda m: float(np.sum(-np.expm1(m*np.log1p(-p))))
    value = invert_expected_unique(expectation, 2.4, 100)["draws"]
    assert expectation(value-1) < 2.4 <= expectation(value)


def test_head_and_workspace_leave_nonempty_complement():
    p = torch.tensor([.5,.1,.1,.1,.05,.05,.05,.05], dtype=torch.float64)
    plan = choose_fixed_sources(p, 4, 1000)
    assert plan["deterministic_sources"] == 1
    q = p.numpy().copy(); q[0] = 0; q /= q.sum()
    expected = lambda m: np.sum(1-(1-q)**m)
    m = plan["draws"]
    assert expected(m-1) < 3 <= expected(m)
    capped = choose_fixed_sources(torch.tensor([.999999,.000001]), 2, 16)
    assert capped["memory_capped"] and capped["draws"] == 16
    assert choose_fixed_sources(torch.ones(1), 1, 10)["deterministic_sources"] == 0


def test_bucket_budget_uses_largest_fitting_power_of_two():
    available = 43 << 30
    out = single_sketch_bucket_budget(available, available)
    for stage, stride in (("tail", 160), ("control", 32)):
        b = out[stage+"_buckets"]
        assert stride*b <= available < stride*2*b
    assert out["control_buckets"] == 1 << 30
    with pytest.raises(MemoryError): single_sketch_bucket_budget(1, 1)


@pytest.mark.parametrize("options", [dict(external_partitions=3), dict(minimum_windows=1),
    dict(fine_threshold=2., coarse_threshold=1.), dict(target_standard_error=0),
    dict(memory_fraction=1.5), dict(buckets=7), dict(seed=-1), dict(maximum_windows=5)])
def test_bad_settings_fail_before_cuda(options):
    with pytest.raises(ValueError): PT2Config(**options)


def test_statistical_accounting_includes_control_and_tail():
    controls, tail = [1.,3.], [.1,.2,.5]
    out = estimate(controls, tail, .01)
    expected = np.var(controls, ddof=1)/2 + np.var(tail, ddof=1)/3
    assert out["standard_error"] == math.sqrt(expected)
    assert out["E2_external"] == np.mean(controls)+np.mean(tail)
    assert control_variance_bound([0.], 1 << 18) == 0
    assert control_variance_bound([1.], 1 << 18) > 1/(1 << 18)
    with pytest.raises(ValueError): estimate([1.], [1.], .1)
