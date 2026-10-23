"""
Oracle pour la distribution cible w_J(theta).

Entrée : theta (Tensor)
Sortie : log-densité non normalisée (Tensor)

Ne dépend d'aucune constante de normalisation.
"""

import torch
import numpy as np
from scipy.optimize import brentq
import scipy.integrate as integrate

# --- Adapté de Approche 1/model.py ---

def fisher_information_1d(theta: torch.Tensor) -> torch.Tensor:
    """Information de Fisher, constante = 1 pour le canal AWGN scalaire."""
    return torch.ones_like(theta)

def cost_fn(theta: torch.Tensor) -> torch.Tensor:
    """Fonction de coût (puissance) : c(theta) = theta^2."""
    return theta ** 2


# --- Adapté de Approche 1/lambda_star.py ---

def _M_of_lambda_torch(lam: float, A: float, n_quad: int = 200) -> float:
    """Calcule M(lambda) = E_w[c(theta)] par quadrature de Gauss-Legendre sur [-A, A]."""
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    theta_np = A * nodes
    w_quad_np = A * weights
    
    theta = torch.tensor(theta_np, dtype=torch.float32)
    w_quad = torch.tensor(w_quad_np, dtype=torch.float32)
    
    J_vals = fisher_information_1d(theta)
    c_vals = cost_fn(theta)
    
    log_vals = 0.5 * torch.log(J_vals) - lam * c_vals
    
    log_Z = torch.logsumexp(log_vals + torch.log(w_quad), dim=0)
    
    log_num = torch.logsumexp(log_vals + torch.log(w_quad) + torch.log(torch.abs(c_vals) + 1e-30), dim=0)
    
    return torch.exp(log_num - log_Z).item()

def solve_lambda_star(P: float, A: float, lambda_max: float = 50.0, n_quad: int = 200) -> float:
    """
    Résout M(lambda*) = P par bissection.
    Retourne lambda* = 0 si la contrainte n'est pas active.
    """
    M0 = _M_of_lambda_torch(0.0, A, n_quad)
    if M0 <= P:
        return 0.0

    def f(lam):
        return _M_of_lambda_torch(lam, A, n_quad) - P

    f_max = f(lambda_max)
    while f_max > 0:
        lambda_max *= 2
        f_max = f(lambda_max)
        if lambda_max > 1e6:
            raise RuntimeError("lambda_max trop grand : vérifier le modèle.")

    return brentq(f, 1e-10, lambda_max, xtol=1e-10)


# --- Fonctions principales ---

def log_w_J_1d(theta: torch.Tensor, lambda_star: float) -> torch.Tensor:
    """
    log-densité non normalisée w_J pour le cas 1D.
    theta: Tensor de shape (..., 1) ou (...)
    """
    J_vals = fisher_information_1d(theta)
    c_vals = cost_fn(theta)
    return 0.5 * torch.log(J_vals) - lambda_star * c_vals

def log_w_J_2d(theta: torch.Tensor, lambda_star: float) -> torch.Tensor:
    """
    log-densité non normalisée w_J pour le cas 2D (produit indépendant de deux cas 1D).
    theta: Tensor de shape (..., 2)
    """
    theta_1 = theta[..., 0]
    theta_2 = theta[..., 1]
    
    log_w_1 = log_w_J_1d(theta_1, lambda_star)
    log_w_2 = log_w_J_1d(theta_2, lambda_star)
    
    return log_w_1 + log_w_2

def theoretical_jeffreys_pdf_1d(theta: torch.Tensor, lambda_star: float) -> torch.Tensor:
    """
    Fonction de référence pour validation : exp(log_w_J_1d) normalisée numériquement
    via scipy.integrate.quad sur le domaine complet.
    """
    def unnorm_pdf(x):
        t = torch.tensor(x, dtype=torch.float32)
        return torch.exp(log_w_J_1d(t, lambda_star)).item()
    
    Z, _ = integrate.quad(unnorm_pdf, -np.inf, np.inf)
    
    res = torch.exp(log_w_J_1d(theta, lambda_star)) / Z
    return res
