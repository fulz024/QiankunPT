import numpy as np
import torch
from qiankunpt import internal_correction


def test_complex_projected_correction_against_constrained_linear_solve():
    h = np.array([[-2., .3+.2j, -.1], [.3-.2j, 1., .4j], [-.1, -.4j, 3.]])
    c = np.array([.8+.2j, .3-.1j, -.4j])
    x = c/np.linalg.norm(c)
    energy = np.vdot(x, h@x).real
    residual = h@x-energy*x
    diagonal = np.diag(h).real
    block = np.block([[np.diag(energy-diagonal), -x[:,None]], [x.conj()[None,:], np.zeros((1,1))]])
    direct = np.linalg.solve(block, np.r_[residual, 0])[:3]
    out = internal_correction(c, h@c, diagonal)
    np.testing.assert_allclose(out["E_S"], energy, atol=1e-14)
    np.testing.assert_allclose(out["E2_internal_projected"], np.vdot(residual, direct).real, atol=1e-13)
    torch.testing.assert_close(out["E2_internal_raw"]+out["E2_Olsen"], out["E2_internal_projected"])
    assert out["correction_orthogonality"] < 1e-13
