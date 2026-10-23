import numpy as np
import pytest

from blahut_arimoto import blahut_arimoto, mutual_information_uniform
from lambda_star import M_of_lambda, find_lambda_star
from model import J_awgn, c_awgn

A = 10.0


@pytest.mark.parametrize("n", [2, 4, 16])
def test_blahut_arimoto_noiseless_channel(n):
    C, p_x = blahut_arimoto(np.eye(n))
    assert C == pytest.approx(np.log2(n), abs=1e-6)
    assert np.allclose(p_x, 1.0 / n)


def test_blahut_arimoto_binary_symmetric_channel():
    p = 0.1
    W = np.array([[1 - p, p], [p, 1 - p]])
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    C, _ = blahut_arimoto(W)
    assert C == pytest.approx(1 - h, abs=1e-6)
    assert mutual_information_uniform(W) == pytest.approx(1 - h, abs=1e-9)


def test_lambda_star_satisfies_power_constraint():
    P = A ** 2 / 9
    lam = find_lambda_star(J_awgn, c_awgn, P, A)
    assert M_of_lambda(lam, J_awgn, c_awgn, A) == pytest.approx(P, rel=1e-6)


def test_lambda_star_decreases_with_power():
    lams = [find_lambda_star(J_awgn, c_awgn, P, A) for P in (2.0, 5.0, 11.0, 20.0)]
    assert all(l1 > l2 for l1, l2 in zip(lams, lams[1:]))
