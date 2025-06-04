import numpy as np
from utils import spod_function


def test_spod_function_simple():
    qhat = np.array([[1.0], [0.0]], dtype=complex)
    w = np.ones((2, 1))
    phi, lam, psi = spod_function(qhat, nblocks=1, dst=1.0, w=w, return_psi=True)
    assert phi.shape == (2, 1)
    assert psi.shape == (1, 1)
    assert np.allclose(lam, 1.0)
    assert np.allclose(phi[:, 0], [1.0, 0.0])

