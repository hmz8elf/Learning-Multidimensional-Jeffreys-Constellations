"""
Modèle de canal — cas jouet AWGN scalaire (papier, §II-D).

Canal : y = theta + z,  z ~ N(0,1)
J(theta) = 1                (constante, Fisher information)
c(theta) = theta^2          (contrainte de puissance moyenne)

Ce cas sert de test unitaire : tout est calculable analytiquement
(le prior de Jeffreys tilté est une Gaussienne tronquée), donc on peut
valider chaque brique de l'algorithme contre une réponse connue.
"""
import numpy as np


def J_awgn(theta: np.ndarray) -> np.ndarray:
    """Information de Fisher, constante = 1 pour le canal AWGN scalaire."""
    return np.ones_like(np.atleast_1d(theta), dtype=float)


def c_awgn(theta: np.ndarray) -> np.ndarray:
    """Fonction de coût (puissance) : c(theta) = theta^2."""
    return np.asarray(theta, dtype=float) ** 2


def log_w_unnormalized(theta: np.ndarray, lam: float, J_func, c_func) -> np.ndarray:
    """
    log [ sqrt(J(theta)) * 2^{-lam * c(theta)} ]  (non normalisé)
    = 0.5*log(J(theta)) - lam*ln(2)*c(theta)
    """
    theta = np.asarray(theta, dtype=float)
    J_vals = J_func(theta)
    c_vals = c_func(theta)
    return 0.5 * np.log(J_vals) - lam * np.log(2) * c_vals
