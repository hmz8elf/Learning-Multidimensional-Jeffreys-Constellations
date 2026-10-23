# -*- coding: utf-8 -*-
"""
validate_step4.py
=====================================================================
Étape 4 — MIMO avec CSI imparfait (III-D), cas isotrope
=====================================================================

Seul cas vraiment multidimensionnel ET couplé du papier (J(theta)
contient un terme theta.theta^T, pas seulement une matrice diagonale
séparable comme le jouet des Étapes 1-3).

Modèle (cf. papier, section III-D, cas isotrope Gamma = ((1-sigma^2)/2) I) :

    theta = [Re(x); Im(x)] in R^{2 nt},  Theta = {theta : ||theta|| <= A}
    c(theta) = ||theta||^2
    J(theta) = 2/(1+sigma^2||theta||^2) * Gamma
               + 4 sigma^4/(1+sigma^2||theta||^2)^2 * theta theta^T

Réduction radiale fermée donnée par le papier (isotropie) :

    p(r) ∝ 2^{-lambda* r^2} * (2(1-sigma^2)/(1+sigma^2 r^2))^{nt}
           * sqrt(1 + 2 sigma^4/(1-sigma^2) * r^2/(1+sigma^2 r^2))
           * r^{2 nt - 1},   r >= 0

C'est notre référence indépendante : on calcule lambda* et la CDF
radiale théorique par quadrature 1D (scipy), et on compare à la norme
||theta|| des points produits par le pipeline général (Cartésien,
domaine "ball", couplage complet).

Réduction d'échelle : le papier utilise nt=4 (d=8). La quadrature
tensorielle de `jeffreys_baseline.py` est documentée pour d <= 4
(cf. docstring du module) ; on utilise donc nt=2 (d=4) ici, ce qui
reste un test multidimensionnel couplé non trivial tout en restant
calculable en un temps raisonnable. Noté explicitement dans les
résultats.
"""

import time
import numpy as np
from scipy import integrate, optimize
from scipy.stats import kstest

from jeffreys_baseline import BaselineConfig, JeffreysConstellationBuilder


# =====================================================================
# Modèle Cartésien (III-D, cas isotrope)
# =====================================================================

def make_csi_model(nt: int, sigma2: float):
    d = 2 * nt
    gamma_scalar = (1.0 - sigma2) / 2.0

    def J_func(theta: np.ndarray) -> np.ndarray:
        norm2 = float(np.dot(theta, theta))
        denom = 1.0 + sigma2 * norm2
        Gamma = gamma_scalar * np.eye(d)
        return (2.0 / denom) * Gamma + (4.0 * sigma2 ** 2 / denom ** 2) * np.outer(theta, theta)

    def c_func(theta: np.ndarray) -> float:
        return float(np.dot(theta, theta))

    return J_func, c_func, d


# =====================================================================
# Référence radiale fermée (indépendante du pipeline)
# =====================================================================

def radial_unnorm_density(r, lam, nt, sigma2):
    denom = 1.0 + sigma2 * r ** 2
    base = (2.0 * (1.0 - sigma2) / denom) ** nt
    correction = np.sqrt(1.0 + (2.0 * sigma2 ** 2 / (1.0 - sigma2)) * (r ** 2 / denom))
    return np.exp(-lam * r ** 2 * np.log(2.0)) * base * correction * r ** (2 * nt - 1)


def reference_lambda_star_radial(A, P, nt, sigma2, lambda_max=50.0):
    def M(lam):
        num, _ = integrate.quad(lambda r: r ** 2 * radial_unnorm_density(r, lam, nt, sigma2),
                                 0.0, A, limit=200)
        den, _ = integrate.quad(lambda r: radial_unnorm_density(r, lam, nt, sigma2),
                                 0.0, A, limit=200)
        return num / den

    m0 = M(0.0)
    if m0 <= P:
        return 0.0
    return optimize.brentq(lambda lam: M(lam) - P, 0.0, lambda_max, xtol=1e-10)


def radial_cdf(A, lam_star, nt, sigma2, n_grid=4000):
    grid = np.linspace(1e-9, A, n_grid)
    unnorm = radial_unnorm_density(grid, lam_star, nt, sigma2)
    cdf_unnorm = np.concatenate([[0.0], np.cumsum(0.5 * (unnorm[1:] + unnorm[:-1]) * np.diff(grid))])
    cdf = cdf_unnorm / cdf_unnorm[-1]

    def F(r):
        return np.interp(r, grid, cdf)

    return F


# =====================================================================
# Exécution
# =====================================================================

def run(A=10.0, sigma2=0.1, nt=2, M_points=6, p=6, n_quad_marg=8, n_quad_fit=40):
    P = A ** 2 / 9.0
    print("=" * 70)
    print(f"III-D CSI imparfait (nt={nt}, d={2*nt}, A={A}, sigma^2={sigma2}, P={P:.4f})")
    print("=" * 70)

    # --- Référence radiale indépendante ---
    lam_ref = reference_lambda_star_radial(A, P, nt, sigma2)
    print(f"lambda* (référence radiale, scipy) = {lam_ref:.4f}")

    # --- Pipeline général (Cartésien, domaine ball, couplage complet) ---
    J_func, c_func, d = make_csi_model(nt, sigma2)
    config = BaselineConfig(A=A, d=d, M=M_points, p=p,
                             n_quad_marg=n_quad_marg, n_quad_fit=n_quad_fit,
                             domain="ball")
    builder = JeffreysConstellationBuilder(J_func=J_func, c_func=c_func, P=P, config=config)

    t0 = time.perf_counter()
    lam_pipeline = builder.compute_lambda_star(verbose=False)
    print(f"lambda* (pipeline, Cartésien d={d})  = {lam_pipeline:.4f}")
    print(f"écart lambda*                       = {abs(lam_pipeline - lam_ref):.2e}")

    X = builder.build(verbose=False)
    dt = time.perf_counter() - t0
    print(f"Temps total (compute_lambda_star + build) : {dt:.1f} s")
    print(f"Taille de la constellation : {X.shape[0]} points en dimension {d} "
          f"(M^d = {M_points}^{d} = {M_points**d})")

    # --- Vérification domaine : tous les points dans la boule ---
    norms = np.linalg.norm(X, axis=1)
    n_out = int((norms > A + 1e-6).sum())
    print(f"Points hors boule (tolérance 1e-6)  : {n_out} / {len(norms)}")

    # --- Comparaison à la référence radiale (KS test) ---
    F_ref = radial_cdf(A, lam_ref, nt, sigma2)
    ks_stat, ks_pvalue = kstest(norms, F_ref)
    print(f"KS statistic (||theta|| vs p(r) théorique) = {ks_stat:.4f} (p-value = {ks_pvalue:.4f})")
    print(f"E[c(theta)] empirique = {np.mean(norms**2):.4f}  (cible P = {P:.4f})")
    print()


if __name__ == "__main__":
    for sigma2 in [0.5, 0.1]:
        run(A=10.0, sigma2=sigma2, nt=2, M_points=6, p=6, n_quad_marg=10)

    print("=" * 70)
    print("CONSTAT (cf. discussion) : lambda* est correct à ~2-3% près, mais")
    print("le test KS échoue nettement (0.27-0.33). Diagnostic : la queue de")
    print("la distribution (r proche de A) est sous-représentée par la grille")
    print("de quantiles deterministe -- max(r) observe ~4.8-5.2 alors que")
    print("A=10, meme en passant M de 6 a 10 (KS ameliore 0.27->0.17 mais")
    print("max(r) ne bouge presque pas). C'est un second mur, statistique,")
    print("distinct du mur de calcul Theta(M^d) : atteindre une combinaison")
    print("JOINTE extreme en dimension couplee est un evenement de probabilite")
    print("tres faible, qu'une grille de quantiles 1D par coordonnee ne peut")
    print("representer qu'avec un M tres grand -- ce qui retombe sur le mur")
    print("de calcul. A documenter comme limite d'Approach I en dimension")
    print("couplee (contrairement au cas separable des Etapes 1-3, ou la")
    print("grille de quantiles est exacte quel que soit M).")
    print("=" * 70)