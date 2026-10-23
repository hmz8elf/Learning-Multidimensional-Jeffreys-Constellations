# -*- coding: utf-8 -*-
"""
validate_step3.py
=====================================================================
Étape 3 — Cas pont : Complex AWGN à détection d'énergie (III-C)
=====================================================================

Ce cas est qualitativement différent des précédents (III-A, III-B,
III-F) : J(theta) n'a PAS de forme fermée. Il nécessite lui-même une
intégration numérique sur la variable de sortie ytilde (loi du
chi-deux non centrale à 2 degrés de liberté, paramètre de
non-centralité 2*theta^2) :

    p(ytilde | theta) = (1/2) exp(-(ytilde + 2 theta^2)/2)
                          I0( sqrt(2 theta^2 ytilde) )

    J(theta) = E_{ytilde|theta} [ ( -2 theta
                    + sqrt(2 ytilde) I1(z)/I0(z) )^2 ],
               z := theta * sqrt(2 ytilde)

Point numérique clé : I0, I1 explosent exponentiellement pour z
grand, mais leur RATIO reste borné. On utilise donc les versions à
échelle exponentielle scipy.special.ive(v, z) = I_v(z) * exp(-|z|),
de sorte que I1(z)/I0(z) = ive(1,z)/ive(0,z) reste stable, sans
jamais évaluer I0/I1 bruts.

Point structurel : le domaine est Theta = [0, A] (theta = |x| >= 0),
pas [-A, A] comme le reste du pipeline suppose. On applique un
changement de variable theta' = theta - A/2 in [-A/2, A/2] : le
pipeline voit un problème symétrique standard, et on ré-évalue
J_func/c_func en theta = theta' + A/2 à l'intérieur du wrapper. Pas
besoin de modifier jeffreys_baseline.py.

Référence : le papier donne explicitement (Fig. 4, P = A^2/9) :
    A = 1.0  -> lambda* = 12.00
    A = 5.0  -> lambda* = 0.34
    A = 10.0 -> lambda* = 0.07
"""

import numpy as np
from scipy import integrate, optimize
from scipy.special import ive

from jeffreys_baseline import BaselineConfig, JeffreysConstellationBuilder


# =====================================================================
# J(theta) pour le canal à détection d'énergie (III-C)
# =====================================================================

def energy_detection_J(theta: float, y_cutoff_sigma: float = 12.0) -> float:
    """
    J(theta) = E_{ytilde|theta} [ (-2 theta + sqrt(2 ytilde) I1(z)/I0(z))^2 ]

    Intégration numérique sur ytilde in [0, +inf), tronquée à
    mean + y_cutoff_sigma * std du chi-deux non central (df=2,
    lambda_nc=2 theta^2) pour rester dans un intervalle fini sans
    perte de masse significative.
    """
    if theta < 1e-10:
        # theta = 0 : z = 0 pour tout ytilde, I1(0)/I0(0) = 0 -> intégrande = 0
        return 0.0

    lam_nc = 2.0 * theta ** 2
    mean = 2.0 + lam_nc
    std = np.sqrt(2.0 * (2.0 + 2.0 * lam_nc))
    y_max = mean + y_cutoff_sigma * std

    def integrand(ytilde):
        z = theta * np.sqrt(2.0 * ytilde)
        ratio = ive(1, z) / ive(0, z)  # stable même si I0, I1 débordent
        bracket = -2.0 * theta + np.sqrt(2.0 * ytilde) * ratio
        # p(ytilde|theta) en forme stable (I0(z) = ive(0,z) * exp(z)) :
        log_p = -0.5 * (ytilde + lam_nc) + z + np.log(ive(0, z))
        return np.exp(log_p) * 0.5 * bracket ** 2

    val, _ = integrate.quad(integrand, 0.0, y_max, limit=200)
    return max(val, 1e-12)


def c_func_energy(theta: float) -> float:
    return theta ** 2


# =====================================================================
# Référence indépendante : lambda* par bissection sur Theta = [0, A]
# =====================================================================

def reference_lambda_star_0A(J_of_theta, c_of_theta, A, P, lambda_max=50.0):
    ln2 = np.log(2.0)

    def M(lam):
        def num_integrand(t):
            return c_of_theta(t) * np.exp(-lam * c_of_theta(t) * ln2) * np.sqrt(J_of_theta(t))

        def den_integrand(t):
            return np.exp(-lam * c_of_theta(t) * ln2) * np.sqrt(J_of_theta(t))

        num, _ = integrate.quad(num_integrand, 0.0, A, limit=200)
        den, _ = integrate.quad(den_integrand, 0.0, A, limit=200)
        return num / den

    m0 = M(0.0)
    if m0 <= P:
        return 0.0, m0
    lam_star = optimize.brentq(lambda lam: M(lam) - P, 0.0, lambda_max, xtol=1e-10)
    return lam_star, M(lam_star)


# =====================================================================
# Wrapper de changement de variable [0,A] -> [-A/2, A/2]
# =====================================================================

def make_shifted_pipeline(J_scalar, c_scalar, A):
    """
    Construit (J_func, c_func, A_eff) pour le pipeline standard
    (domaine [-A_eff, A_eff]), correspondant au problème original sur
    Theta = [0, A] via theta = theta_shifted + A/2.
    """
    A_eff = A / 2.0

    def J_func(theta_vec):
        theta = theta_vec[0] + A_eff
        return np.array([[J_scalar(theta)]])

    def c_func(theta_vec):
        theta = theta_vec[0] + A_eff
        return c_scalar(theta)

    return J_func, c_func, A_eff


def run_case(A, P_ratio=1.0 / 9.0, lambda_paper=None, M_points=200, p=6, n_quad_fit=60):
    P = A ** 2 * P_ratio
    print("=" * 70)
    print(f"III-C Détection d'énergie (A={A}, P={P:.4f})")
    print("=" * 70)

    # --- Référence indépendante sur Theta=[0,A] ---
    lam_ref, m_ref = reference_lambda_star_0A(energy_detection_J, c_func_energy, A, P)
    print(f"lambda* (référence scipy)        = {lam_ref:.4f}")
    if lambda_paper is not None:
        print(f"lambda* (papier, Fig. 4)          = {lambda_paper:.2f}")

    # --- Pipeline général, via le changement de variable ---
    J_func, c_func, A_eff = make_shifted_pipeline(energy_detection_J, c_func_energy, A)
    config = BaselineConfig(A=A_eff, d=1, M=M_points, p=p,
                             n_quad_marg=8, n_quad_fit=n_quad_fit)
    builder = JeffreysConstellationBuilder(J_func=J_func, c_func=c_func, P=P, config=config)
    lam_pipeline = builder.compute_lambda_star(verbose=False)
    print(f"lambda* (pipeline, via shift)     = {lam_pipeline:.4f}")
    print(f"écart pipeline vs référence       = {abs(lam_pipeline - lam_ref):.2e}")
    if lambda_paper is not None:
        print(f"écart pipeline vs papier          = {abs(lam_pipeline - lambda_paper):.2e}")

    X = builder.build(verbose=False)
    theta_original = X[:, 0] + A_eff  # ramène dans [0, A] pour interprétation
    mean_c = np.mean([c_func_energy(t) for t in theta_original])
    print(f"E[c(theta)] empirique             = {mean_c:.4f}  (cible P = {P:.4f})")
    print(f"theta_original dans [0, A] ?      = "
          f"{'oui' if theta_original.min() >= -1e-6 and theta_original.max() <= A + 1e-6 else 'NON -> bug'}")
    print()


if __name__ == "__main__":
    run_case(A=1.0, lambda_paper=12.00)
    run_case(A=5.0, lambda_paper=0.34)
    run_case(A=10.0, lambda_paper=0.07)
