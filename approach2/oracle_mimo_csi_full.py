"""
oracle_mimo_csi_full.py
======================
Oracle MIMO CSI imparfait COMPLET (Section III-D, Yang & Combes 2025).

Modèle général :
  y = H * x + noise,  H = H_hat + H_tilde
  x = theta (complexe reécrit en réel : theta = [Re(x); Im(x)] in R^{2*n_t})
  - Les lignes de H_hat ~ p(h_hat) avec covariance Gamma_complex (n_t x n_t complexe)
  - Les entrées de H_tilde ~ CN(0, sigma2) : erreur d'estimation CSI
  - denom = 1 + sigma2 * ||theta||^2

Formule de la matrice de Fisher (éq. du papier, §III-D, forme générale) :
  J(theta) = 2/denom * Gamma + 4*sigma2^2/denom^2 * theta*theta^T  [d x d, d=2*n_t]

  où Gamma = E[h_hat * h_hat^H] dans l'espace réel 2n_t x 2n_t, i.e. :
  pour h_hat complexe de covariance Gamma_complex (n_t x n_t),
  la représentation réelle donne :
      Gamma_real = [[Re(Gamma_complex), -Im(Gamma_complex)],
                    [Im(Gamma_complex),  Re(Gamma_complex)]]

Cas isotrope (Semaine 4 précédente) :
  Gamma_complex = ((1 - sigma2) / 2) * I_{n_t}
  -> Gamma_real = ((1 - sigma2) / 2) * I_{2n_t}

Cas NON-isotrope (cette implémentation) :
  Gamma_complex quelconque (e.g. diagonale avec valeurs propres distinctes),
  traduit en Gamma_real par le bloc ci-dessus.

NOTES MATHÉMATIQUES :
  La dérivation complète de J(theta) suit la même chaîne que dans §III-D :
  - La vraisemblance conjointe (y, h_hat | theta) se factorise en produit
    sur n_r antennes (i.i.d.).
  - Par la règle d'information de Fisher pour des mélanges gaussiens,
    la contribution de chaque antenne donne :
      J_r(theta) = 2 * Gamma * (1/denom) + 4 * sigma2^2 * theta*theta^T / denom^2
    (cf. calcul par complétion de carré dans la représentation réelle, eq.(12)
    et dérivation suivante dans le papier).
  - La matrice Fisher totale sur n_r antennes est n_r * J_r(theta), mais
    comme dans notre framework on normalise par n_r, J(theta) = J_r(theta).
  - La différence avec le cas isotrope est UNIQUEMENT dans Gamma :
    cas isotrope  : Gamma = gamma * I   (scalaire)
    cas général   : Gamma = Gamma_real  (matrice pleine)
  - J(theta) reste différentiable et définie positive (sous Gamma > 0)
    => torch.linalg.slogdet est stable.

Stabilité numérique :
  - Si sigma2 ~ 0, le terme 4*sigma2^2/denom^2 disparaît, J -> 2*Gamma/1 = 2*Gamma.
  - Si ||theta|| -> A grand, denom -> sigma2*A^2, J reste borné.
  - La positivité est garantie si Gamma > 0 (définie positive).
"""

import torch
import numpy as np
from scipy.optimize import brentq
import scipy.integrate as integrate


def make_gamma_real(gamma_complex: np.ndarray) -> torch.Tensor:
    """
    Convertit une matrice de covariance complexe Gamma_complex (n_t x n_t)
    en sa représentation réelle (2n_t x 2n_t) :
      Gamma_real = [[Re(Gamma), -Im(Gamma)],
                    [Im(Gamma),  Re(Gamma)]]
    """
    R = np.real(gamma_complex)
    I = np.imag(gamma_complex)
    top = np.concatenate([R, -I], axis=1)
    bot = np.concatenate([I,  R], axis=1)
    return torch.tensor(np.concatenate([top, bot], axis=0), dtype=torch.float32)


def fisher_information_mimo_full(
    theta: torch.Tensor,
    sigma2: float,
    Gamma_real: torch.Tensor,
) -> torch.Tensor:
    """
    Information de Fisher complète pour MIMO CSI imparfait (Section III-D).

    theta      : (batch, d) avec d = 2*n_t
    sigma2     : variance de l'erreur d'estimation H_tilde ~ CN(0, sigma2)
    Gamma_real : (d, d) matrice de covariance réelle de h_hat (cf. make_gamma_real)

    Retourne J de shape (batch, d, d).
    """
    batch = theta.shape[0]
    d = theta.shape[1]
    device = theta.device

    Gamma_real = Gamma_real.to(device)

    norm2 = torch.sum(theta ** 2, dim=-1)           # (batch,)
    denom = 1.0 + sigma2 * norm2                    # (batch,)

    # Terme 1 : 2/denom * Gamma    (shape batch x d x d)
    coef1 = 2.0 / denom                              # (batch,)
    term1 = coef1.view(-1, 1, 1) * Gamma_real.unsqueeze(0).expand(batch, -1, -1)

    # Terme 2 : 4*sigma2^2/denom^2 * theta*theta^T  (shape batch x d x d)
    coef2 = 4.0 * sigma2 ** 2 / (denom ** 2)         # (batch,)
    outer = torch.bmm(theta.unsqueeze(-1), theta.unsqueeze(-2))   # (batch, d, d)
    term2 = coef2.view(-1, 1, 1) * outer

    return term1 + term2


def log_w_J_mimo_full(
    theta: torch.Tensor,
    lambda_star: float,
    sigma2: float,
    Gamma_real: torch.Tensor,
) -> torch.Tensor:
    """
    log-densité non normalisée :
      log w_J(theta) = 0.5 * logdet(J(theta)) - lambda_star * ||theta||^2
    """
    J = fisher_information_mimo_full(theta, sigma2, Gamma_real)
    sign, logdet = torch.linalg.slogdet(J)
    c = torch.sum(theta ** 2, dim=-1)
    return 0.5 * logdet - lambda_star * c


# ------------------------------------------------------------------ #
#  Résolution de lambda* par intégration Monte-Carlo pour le cas
#  non-isotrope (pas de formule radiale fermée).
# ------------------------------------------------------------------ #

def solve_lambda_star_mc(
    P: float,
    A: float,
    sigma2: float,
    Gamma_real: torch.Tensor,
    n_mc: int = 50000,
    lambda_max: float = 50.0,
) -> float:
    """
    Résout E_{theta ~ w_J(lambda)}[||theta||^2] = P par bissection.
    L'espérance est estimée par Monte-Carlo uniforme sur la boule ||theta|| <= A.

    Pour le cas isotrope on dispose d'une intégration radiale 1D exacte ;
    ici on utilise MC pour rester générique (coût O(n_mc) indépendant de d).
    """
    d = Gamma_real.shape[0]

    # Tirage uniforme sur la boule de rayon A en dimension d (rejection sampling)
    rng = np.random.default_rng(0)
    theta_np = rng.standard_normal((n_mc, d)).astype(np.float32)
    theta_np /= np.linalg.norm(theta_np, axis=1, keepdims=True)
    radii = rng.random(n_mc) ** (1.0 / d) * A
    theta_np *= radii[:, None]
    theta_t = torch.from_numpy(theta_np)

    norm2_np = np.sum(theta_np ** 2, axis=1)

    def M_of_lambda(lam: float) -> float:
        with torch.no_grad():
            log_w = log_w_J_mimo_full(theta_t, lam, sigma2, Gamma_real)
        log_w_np = log_w.numpy()
        log_w_np -= log_w_np.max()
        w = np.exp(log_w_np)
        w /= w.sum()
        return float(np.dot(w, norm2_np))

    m0 = M_of_lambda(0.0)
    if m0 <= P:
        return 0.0

    return brentq(lambda lam: M_of_lambda(lam) - P,
                  0.0, lambda_max, xtol=1e-8)


# ------------------------------------------------------------------ #
#  Helper : Gamma non-isotrope exemple (diagonale inégale)
# ------------------------------------------------------------------ #

def make_anisotropic_gamma(n_t: int, sigma2: float, ratio: float = 3.0) -> torch.Tensor:
    """
    Construit une Gamma_complex diagonale avec des valeurs propres
    s'étalant d'un facteur `ratio` de la plus grande à la plus petite,
    tout en respectant Tr(Gamma_complex) = n_t * (1 - sigma2) / 2
    (même puissance totale que le cas isotrope).

    Exemple pour n_t=2, ratio=3 : eigenvalues [3k, k] avec k choisi
    pour que leur somme = n_t * (1 - sigma2) / 2.
    """
    total_power = n_t * (1.0 - sigma2) / 2.0
    # Valeurs propres : linspace(ratio, 1, n_t), normalisées à la somme voulue
    eigs = np.linspace(ratio, 1.0, n_t)
    eigs = eigs / eigs.sum() * total_power
    gamma_complex = np.diag(eigs)
    return make_gamma_real(gamma_complex)
