# -*- coding: utf-8 -*-
"""
validate_step2.py
=====================================================================
Étape 2 — Cas scalaires du papier (Yang & Combes, arXiv:2504.13325)
=====================================================================

Pour chaque canal scalaire à domaine symétrique Θ = [-A, A] :

  III-A : AWGN avec clipping
  III-B : AWGN avec ADC bas-résolution (L niveaux)
  III-F : 1-bit ADC avec dithering

on compare deux choses, indépendamment l'une de l'autre :

  1. lambda* calculé par `JeffreysConstellationBuilder` (pipeline
     général : quadrature tensorielle + bissection, cf.
     `compute_lambda_star`) ;
  2. lambda*_ref calculé par une implémentation de référence
     indépendante (scipy.integrate.quad + scipy.optimize.brentq),
     qui ne partage aucun code avec le pipeline.

Si les deux coïncident, `compute_lambda_star` est validé sur des cas
non triviaux (pas seulement le jouet séparable de l'Étape 1). On
vérifie aussi que la constellation finale (histogramme empirique)
suit bien la densité théorique wJ,lambda*_ref via un test de
Kolmogorov-Smirnov.

Note : le cas Poisson (III-E) et le cas CSI imparfait (III-D, radial)
ont un domaine Theta = [0, A], non supporté par `BaselineConfig`
(qui suppose Theta = [-A, A]^d). Mis de côté ici, à traiter en
généralisant le code (cf. remarque de fin de fichier).
"""

import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm, kstest

from jeffreys_baseline import BaselineConfig, JeffreysConstellationBuilder

Q = norm.sf   # fonction de survie = Q(x) = P(Z > x)
phi = norm.pdf


# =====================================================================
# Implémentation de référence indépendante (scipy), commune aux 3 cas
# =====================================================================

def reference_lambda_star(J_of_theta, c_of_theta, A, P, lambda_max=50.0):
    """
    Résout M(lambda) = P par bissection (brentq), où

        M(lambda) = E_theta~wJ,lambda [c(theta)]
                  = int c(t) 2^{-lambda c(t)} sqrt(J(t)) dt
                    / int 2^{-lambda c(t)} sqrt(J(t)) dt

    Les deux intégrales sont calculées par quadrature adaptative
    (scipy.integrate.quad), indépendamment de la quadrature
    Gauss-Legendre tensorielle utilisée dans jeffreys_baseline.py.
    """
    ln2 = np.log(2.0)

    def M(lam):
        def num_integrand(t):
            return c_of_theta(t) * np.exp(-lam * c_of_theta(t) * ln2) * np.sqrt(J_of_theta(t))

        def den_integrand(t):
            return np.exp(-lam * c_of_theta(t) * ln2) * np.sqrt(J_of_theta(t))

        num, _ = integrate.quad(num_integrand, -A, A, limit=200)
        den, _ = integrate.quad(den_integrand, -A, A, limit=200)
        return num / den

    m0 = M(0.0)
    if m0 <= P:
        return 0.0, m0
    lam_star = optimize.brentq(lambda lam: M(lam) - P, 0.0, lambda_max, xtol=1e-10)
    return lam_star, M(lam_star)


def reference_cdf(J_of_theta, c_of_theta, A, lam_star, n_grid=4000):
    """
    Construit la CDF théorique F(theta) = int_{-A}^{theta} wJ,lambda*
    par intégration numérique sur une grille fine, pour comparaison
    KS avec l'histogramme empirique de la constellation.
    Retourne un callable F(theta) (interpolation linéaire).
    """
    ln2 = np.log(2.0)
    grid = np.linspace(-A, A, n_grid)
    unnorm = np.array([
        np.exp(-lam_star * c_of_theta(t) * ln2) * np.sqrt(J_of_theta(t))
        for t in grid
    ])
    cdf_unnorm = np.concatenate([[0.0], np.cumsum(
        0.5 * (unnorm[1:] + unnorm[:-1]) * np.diff(grid))])
    cdf = cdf_unnorm / cdf_unnorm[-1]

    def F(theta):
        return np.interp(theta, grid, cdf)

    return F


def run_case(name, J_func, c_func, A, P, M_points=300, p=6, n_quad_fit=60):
    print("=" * 70)
    print(name)
    print("=" * 70)

    # --- 1. lambda* : pipeline général vs référence indépendante ---
    config = BaselineConfig(A=A, d=1, M=M_points, p=p,
                             n_quad_marg=8, n_quad_fit=n_quad_fit)
    builder = JeffreysConstellationBuilder(
        J_func=lambda th: np.array([[J_func(th[0])]]),
        c_func=lambda th: c_func(th[0]),
        P=P, config=config)
    lam_pipeline = builder.compute_lambda_star(verbose=False)

    lam_ref, m_ref = reference_lambda_star(J_func, c_func, A, P)

    print(f"lambda* (pipeline)  = {lam_pipeline:.6f}")
    print(f"lambda* (référence) = {lam_ref:.6f}")
    print(f"écart absolu        = {abs(lam_pipeline - lam_ref):.2e}")

    # --- 2. constellation complète + test KS contre la CDF théorique ---
    X = builder.build(verbose=False)
    theta_samples = X[:, 0]

    F_ref = reference_cdf(J_func, c_func, A, lam_ref)
    ks_stat, ks_pvalue = kstest(theta_samples, F_ref)
    print(f"KS statistic         = {ks_stat:.4f}  (p-value = {ks_pvalue:.4f})")
    print(f"E[c(theta)] empirique = {np.mean([c_func(t) for t in theta_samples]):.4f}  (cible P = {P:.4f})")
    print()

    return lam_pipeline, lam_ref, ks_stat


# =====================================================================
# III-A : AWGN avec clipping
# =====================================================================

def clipping_case(alpha, A=5.0):
    B = alpha * A

    def J_func(theta):
        s1, s2 = B + theta, B - theta
        val = 1.0
        for s in (s1, s2):
            qs = Q(s)
            if qs < 1e-300:
                continue  # terme négligeable, évite une division par ~0
            val -= (qs + s * phi(s) - phi(s) ** 2 / qs)
        return max(val, 1e-12)

    def c_func(theta):
        return theta ** 2

    return J_func, c_func, A


# =====================================================================
# III-B : AWGN avec ADC bas-résolution (L niveaux)
# =====================================================================

def adc_case(L, A=5.0):
    if L == 2:
        thresholds = np.array([0.0])
    else:
        thresholds = np.linspace(-A, A, L - 1)
    t = np.concatenate([[-np.inf], thresholds, [np.inf]])

    def J_func(theta):
        total = 0.0
        for ell in range(1, L + 1):
            t_lo, t_hi = t[ell - 1], t[ell]
            phi_lo = phi(theta - t_lo) if np.isfinite(t_lo) else 0.0
            phi_hi = phi(theta - t_hi) if np.isfinite(t_hi) else 0.0
            Q_lo = Q(theta - t_lo) if np.isfinite(t_lo) else 0.0   # Q(theta-(-inf)) = Q(+inf) = 0
            Q_hi = Q(theta - t_hi) if np.isfinite(t_hi) else 1.0   # Q(theta-(+inf)) = Q(-inf) = 1
            denom = Q_hi - Q_lo  # p(bin) = Q(theta-t_hi) - Q(theta-t_lo) >= 0 (eq. 11 du papier)
            if denom < 1e-300:
                continue
            total += (phi_lo - phi_hi) ** 2 / denom
        return max(total, 1e-12)

    def c_func(theta):
        return theta ** 2

    return J_func, c_func, A


# =====================================================================
# III-F : 1-bit ADC avec dithering
# =====================================================================

def dithering_case(alpha, N=3, A=10.0):
    s_points = np.linspace(-alpha * A, alpha * A, N)

    def J_func(theta):
        vals = []
        for s in s_points:
            u = theta - s
            qu = Q(u)
            denom = qu * (1.0 - qu)
            if denom < 1e-300:
                continue
            vals.append(phi(u) ** 2 / denom)
        return max(float(np.mean(vals)), 1e-12) if vals else 1e-12

    def c_func(theta):
        return theta ** 2

    return J_func, c_func, A


if __name__ == "__main__":
    P_ratio = 1.0 / 9.0  # P = A^2 / 9, comme dans le papier

    # A=20 confirmé par correspondance exacte avec Fig. 2 du papier
    # (lambda* papier : 0.00, 0.00, 0.02, 0.02 pour alpha=0.1,0.5,1,2)
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        J_func, c_func, A = clipping_case(alpha, A=20.0)
        run_case(f"III-A Clipping (alpha={alpha}, A={A})", J_func, c_func, A, P=A ** 2 * P_ratio)

    # A=5 confirmé par correspondance exacte avec Fig. 3 du papier
    # (lambda* papier : 0.00, 0.20, 0.25, 0.25 pour L=2,4,8,16)
    for L in [2, 4, 8, 16]:
        J_func, c_func, A = adc_case(L, A=5.0)
        run_case(f"III-B ADC (L={L}, A={A})", J_func, c_func, A, P=A ** 2 * P_ratio)

    # A=10, N=3 (valeurs explicites du papier, Fig. 7)
    # (lambda* papier : 0.00, 0.05, 0.04 pour alpha=0,0.5,1)
    for alpha in [0.0, 0.5, 1.0]:
        J_func, c_func, A = dithering_case(alpha, N=3, A=10.0)
        run_case(f"III-F Dithering (alpha={alpha}, N=3, A={A})", J_func, c_func, A, P=A ** 2 * P_ratio)

    print("=" * 70)
    print("Note : cas III-E (Poisson) et III-D (CSI, radial) laissés de côté :")
    print("domaine Theta=[0,A], non supporté par BaselineConfig (Theta=[-A,A]^d).")
    print("=" * 70)
