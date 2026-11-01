"""Projected internal correction for a supplied finite-support Hamiltonian action."""
import math
import numpy as np
import torch


def internal_correction(coefficients, hpsi, diagonal, *, denominator_epsilon=1e-10):
    """Return E_S, the residual, raw internal PT2 and the Olsen correction.

    All three input vectors must describe the SAME ordered finite support.
    hpsi is H_SS @ coefficients, supplied by the caller's solver or backend.
    No eigenproblem is solved and no full-network VMC energy is inferred.
    """
    c = torch.as_tensor(coefficients, dtype=torch.complex128)
    h = torch.as_tensor(hpsi, dtype=torch.complex128, device=c.device)
    if isinstance(diagonal, np.ndarray) and not diagonal.flags.writeable:
        diagonal = diagonal.copy()
    d = torch.as_tensor(diagonal, dtype=torch.float64, device=c.device)
    if c.ndim != 1 or c.numel() == 0 or h.shape != c.shape or d.shape != c.shape:
        raise ValueError("Require three nonempty vectors of the same length")
    if not all(bool(torch.isfinite(x).all()) for x in (c, h, d)):
        raise ValueError("Inputs must be finite")
    if not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0:
        raise ValueError("denominator_epsilon must be finite and positive")
    norm = torch.vdot(c, c).real
    if not bool(torch.isfinite(norm) & (norm > 0)):
        raise ValueError("Require finite positive coefficient norm")
    x, hx = c/norm.sqrt(), h/norm.sqrt()
    energy = torch.vdot(x, hx).real
    residual = hx-energy*x
    residual = residual-x*torch.vdot(x, residual)
    denominator = energy-d
    denominator = torch.where(denominator.abs() < denominator_epsilon,
        torch.where(denominator >= 0, denominator_epsilon, -denominator_epsilon), denominator)
    overlap = torch.vdot(x, residual/denominator)
    weight = torch.vdot(x, x/denominator)
    if bool(weight.abs() <= torch.finfo(torch.float64).eps):
        raise ArithmeticError("Singular projected internal constraint")
    raw = torch.vdot(residual, residual/denominator).real
    correction = (residual-overlap/weight*x)/denominator
    projected = torch.vdot(residual, correction).real
    return dict(E_S=energy, E2_internal_raw=raw, E2_Olsen=projected-raw,
                E2_internal_projected=projected, residual_norm=torch.linalg.vector_norm(residual),
                correction_orthogonality=torch.vdot(x, correction).abs())
