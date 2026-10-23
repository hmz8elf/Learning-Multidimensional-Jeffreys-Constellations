import torch
import numpy as np
from scipy.optimize import brentq
import scipy.integrate as integrate

def fisher_information_nongaussian(theta: torch.Tensor) -> torch.Tensor:
    """Information de Fisher non-constante : J(theta) = 1 + theta^2"""
    return 1.0 + theta**2

def cost_fn(theta: torch.Tensor) -> torch.Tensor:
    """Fonction de coût (puissance) : c(theta) = theta^2."""
    return theta ** 2

def _M_of_lambda_torch_nongaussian(lam: float, A: float, n_quad: int = 200) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    theta_np = A * nodes
    w_quad_np = A * weights
    
    theta = torch.tensor(theta_np, dtype=torch.float32)
    w_quad = torch.tensor(w_quad_np, dtype=torch.float32)
    
    J_vals = fisher_information_nongaussian(theta)
    c_vals = cost_fn(theta)
    
    log_vals = 0.5 * torch.log(J_vals) - lam * c_vals
    
    log_Z = torch.logsumexp(log_vals + torch.log(w_quad), dim=0)
    
    log_num = torch.logsumexp(log_vals + torch.log(w_quad) + torch.log(torch.abs(c_vals) + 1e-30), dim=0)
    
    return torch.exp(log_num - log_Z).item()

def solve_lambda_star_nongaussian(P: float, A: float, lambda_max: float = 50.0, n_quad: int = 200) -> float:
    M0 = _M_of_lambda_torch_nongaussian(0.0, A, n_quad)
    if M0 <= P:
        return 0.0

    def f(lam):
        return _M_of_lambda_torch_nongaussian(lam, A, n_quad) - P

    f_max = f(lambda_max)
    while f_max > 0:
        lambda_max *= 2
        f_max = f(lambda_max)
        if lambda_max > 1e6:
            raise RuntimeError("lambda_max trop grand : vérifier le modèle.")

    return brentq(f, 1e-10, lambda_max, xtol=1e-10)

def log_w_J_nongaussian_1d(theta: torch.Tensor, lambda_star: float) -> torch.Tensor:
    J_vals = fisher_information_nongaussian(theta)
    c_vals = cost_fn(theta)
    return 0.5 * torch.log(J_vals) - lambda_star * c_vals

def theoretical_pdf_nongaussian_1d(theta: torch.Tensor, lambda_star: float) -> torch.Tensor:
    def unnorm_pdf(x):
        t = torch.tensor(x, dtype=torch.float32)
        return torch.exp(log_w_J_nongaussian_1d(t, lambda_star)).item()
    
    Z, _ = integrate.quad(unnorm_pdf, -np.inf, np.inf)
    
    res = torch.exp(log_w_J_nongaussian_1d(theta, lambda_star)) / Z
    return res
