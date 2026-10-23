import numpy as np
import pytest
import scipy.integrate as integrate
import torch

from oracle_mimo import fisher_information_mimo, log_w_J_mimo, solve_lambda_star_mimo_radial
from oracle_mimo_csi_full import log_w_J_mimo_full, make_gamma_real

SIGMA2 = 0.1


@pytest.mark.parametrize("n_t", [1, 2, 4])
def test_fisher_matrix_is_symmetric_positive_definite(n_t):
    torch.manual_seed(0)
    theta = 3.0 * torch.randn(64, 2 * n_t)
    J = fisher_information_mimo(theta, SIGMA2, n_t)
    assert torch.allclose(J, J.transpose(1, 2))
    assert (torch.linalg.eigvalsh(J) > 0).all()


@pytest.mark.parametrize("n_t", [2, 4])
def test_logdet_matches_closed_form(n_t):
    # J = a I + b theta theta^T  =>  det J = a^d (1 + b ||theta||^2 / a)
    torch.manual_seed(1)
    d = 2 * n_t
    theta = 2.0 * torch.randn(32, d, dtype=torch.float64)
    r2 = (theta ** 2).sum(-1)
    denom = 1 + SIGMA2 * r2
    a = (1 - SIGMA2) / denom
    b = 4 * SIGMA2 ** 2 / denom ** 2
    expected = 0.5 * (d * torch.log(a) + torch.log1p(b * r2 / a))
    got = log_w_J_mimo(theta, lambda_star=0.0, sigma2=SIGMA2, n_t=n_t)
    assert torch.allclose(got, expected, atol=1e-10)


def test_full_csi_oracle_reduces_to_isotropic_oracle():
    n_t = 2
    torch.manual_seed(2)
    theta = 2.0 * torch.randn(16, 2 * n_t)
    gamma = make_gamma_real(((1 - SIGMA2) / 2) * np.eye(n_t))
    lam = 0.14
    full = log_w_J_mimo_full(theta, lam, SIGMA2, gamma)
    iso = log_w_J_mimo(theta, lam, SIGMA2, n_t)
    assert torch.allclose(full, iso, atol=1e-5)


@pytest.mark.xfail(strict=True, reason=(
    "Known limitation B1: solve_lambda_star_mimo_radial solves with exp(-lambda*ln2*r^2) "
    "but log_w_J_mimo uses exp(-lambda*r^2), so E||theta||^2 < P under the sampled density."))
@pytest.mark.parametrize("n_t", [2, 4, 8])
def test_sampled_density_satisfies_power_constraint(n_t):
    A, P = 10.0, 10.0 ** 2 / 9
    lam = solve_lambda_star_mimo_radial(P, A, n_t, SIGMA2)

    # Radial density of exp(log_w_J_mimo), with the exact det J and the r^(d-1) Jacobian.
    def radial(r):
        denom = 1 + SIGMA2 * r * r
        a = (1 - SIGMA2) / denom
        b = 4 * SIGMA2 ** 2 / denom ** 2
        return np.exp(-lam * r * r) * a ** n_t * np.sqrt(1 + b * r * r / a) * r ** (2 * n_t - 1)

    num = integrate.quad(lambda r: r * r * radial(r), 0, A, limit=200)[0]
    den = integrate.quad(radial, 0, A, limit=200)[0]
    assert num / den == pytest.approx(P, rel=0.02)
