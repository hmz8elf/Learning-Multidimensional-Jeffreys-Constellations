import torch
import numpy as np
from scipy.optimize import brentq
import scipy.integrate as integrate

def fisher_information_mimo(theta: torch.Tensor, sigma2: float, n_t: int) -> torch.Tensor:
    """
    Information de Fisher pour MIMO CSI imparfait, cas isotrope (Section III-D).
    theta: Tensor de shape (batch_size, d) où d = 2 * n_t.
    Retourne J de shape (batch_size, d, d).
    
    Formule:
    d = 2 * n_t
    Gamma = ((1 - sigma2) / 2) * I
    J(theta) = (2 / denom) * Gamma + (4 * sigma2**2 / denom**2) * outer(theta, theta)
    où denom = 1 + sigma2 * ||theta||^2
    """
    d = 2 * n_t
    gamma_scalar = (1.0 - sigma2) / 2.0
    
    norm2 = torch.sum(theta**2, dim=-1) # (batch_size,)
    denom = 1.0 + sigma2 * norm2 # (batch_size,)
    
    coef1 = 2.0 * gamma_scalar / denom # (batch_size,)
    term1 = coef1.view(-1, 1, 1) * torch.eye(d, device=theta.device).unsqueeze(0)
    
    coef2 = 4.0 * sigma2**2 / (denom**2) # (batch_size,)
    term2 = coef2.view(-1, 1, 1) * torch.bmm(theta.unsqueeze(-1), theta.unsqueeze(-2))
    
    return term1 + term2

def cost_fn_mimo(theta: torch.Tensor) -> torch.Tensor:
    """Contrainte de puissance: ||theta||^2"""
    return torch.sum(theta**2, dim=-1)

def log_w_J_mimo(theta: torch.Tensor, lambda_star: float, sigma2: float, n_t: int) -> torch.Tensor:
    """
    log-densité non normalisée w_J pour le MIMO isotrope.
    """
    J = fisher_information_mimo(theta, sigma2, n_t)
    sign, logdet = torch.linalg.slogdet(J)
    
    c = cost_fn_mimo(theta)
    
    return 0.5 * logdet - lambda_star * c

# Pour résoudre lambda_star en MIMO isotrope, on utilise la méthode de quadrature radiale
# (car l'intégration cartésienne est trop coûteuse en grande dimension).
def radial_unnorm_density(r: float, lam: float, nt: int, sigma2: float) -> float:
    denom = 1.0 + sigma2 * r ** 2
    base = (2.0 * (1.0 - sigma2) / denom) ** nt
    correction = np.sqrt(1.0 + (2.0 * sigma2 ** 2 / (1.0 - sigma2)) * (r ** 2 / denom))
    return np.exp(-lam * r ** 2 * np.log(2.0)) * base * correction * r ** (2 * nt - 1)

def solve_lambda_star_mimo_radial(P: float, A: float, nt: int, sigma2: float, lambda_max: float = 50.0) -> float:
    """
    Résout lambda* pour le cas MIMO isotrope (radiale)
    """
    def M(lam):
        num, _ = integrate.quad(lambda r: r ** 2 * radial_unnorm_density(r, lam, nt, sigma2),
                                 0.0, A, limit=200)
        den, _ = integrate.quad(lambda r: radial_unnorm_density(r, lam, nt, sigma2),
                                 0.0, A, limit=200)
        return num / den

    m0 = M(0.0)
    if m0 <= P:
        return 0.0
        
    def f(lam):
        return M(lam) - P
        
    return brentq(f, 0.0, lambda_max, xtol=1e-10)
