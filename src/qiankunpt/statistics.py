"""Window statistics used by the fixed-parameter production evaluator."""
import math
import numpy as np


def control_variance_bound(norm_values, buckets, failure_probability=1e-3/16):
    """Norm-based variance bound under independent ideal complex sketch maps.

    This is a model bound, not a confidence interval for total energy. The
    seeded implementation and stopping on an observed SE need separate
    empirical coverage checks, especially for rare external determinants.
    """
    values = np.asarray(norm_values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid unsigned sketch norms")
    relative = 1/math.sqrt(buckets*len(values)*failure_probability)
    if relative >= 1:
        raise ValueError("Sketch too small for the control norm bound")
    upper = float(values.mean())/(1-relative)
    return upper*upper/(buckets*len(values))


def estimate(control_values, tail_values, control_variance):
    c, t = np.asarray(control_values, float), np.asarray(tail_values, float)
    if len(c) < 1 or len(t) < 2 or not np.isfinite(c).all() or not np.isfinite(t).all():
        raise ValueError("Need finite control values and two independent tail windows")
    cv = max(float(control_variance), float(c.var(ddof=1)/len(c)) if len(c) > 1 else 0.)
    tv = float(t.var(ddof=1)/len(t))
    return dict(E2_external=float(c.mean()+t.mean()), standard_error=math.sqrt(cv+tv),
                control_estimate=float(c.mean()), tail_estimate=float(t.mean()),
                control_variance=cv, tail_variance=tv)
