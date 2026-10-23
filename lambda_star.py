"""
Étape 1 de l'algorithme : calcul du multiplicateur lambda*.

M(lambda) = E_{theta ~ w_{J,lambda}}[c(theta)]
On résout M(lambda*) = P par bissection (Lemme 1 : M strictement décroissante).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp


def M_of_lambda(lam: float, J_func, c_func, A: float, n_quad: int = 200) -> float:
    """
    Calcule M(lambda) = E_w[c(theta)] par quadrature de Gauss-Legendre sur [-A, A].
    Tout est fait en log-espace pour la stabilité numérique.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    theta = A * nodes            # rescale [-1,1] -> [-A,A]
    w_quad = A * weights

    log_vals = 0.5 * np.log(J_func(theta)) - lam * np.log(2) * c_func(theta)

    # normalisation : log Z = logsumexp(log_vals + log(w_quad))
    log_Z = logsumexp(log_vals + np.log(w_quad))
    # E[c(theta)] = sum( c(theta) * exp(log_vals - log_Z) * w_quad )
    log_num = logsumexp(log_vals + np.log(w_quad) + np.log(np.abs(c_func(theta)) + 1e-300))
    # c(theta) >= 0 ici (c'est theta^2), donc pas de souci de signe
    return np.exp(log_num - log_Z)


def find_lambda_star(J_func, c_func, P: float, A: float,
                      lambda_max: float = 50.0, n_quad: int = 200) -> float:
    """
    Résout M(lambda*) = P par bissection.
    Retourne lambda* = 0 si la contrainte moyenne n'est pas active.
    """
    M0 = M_of_lambda(0.0, J_func, c_func, A, n_quad)
    if M0 <= P:
        return 0.0

    def f(lam):
        return M_of_lambda(lam, J_func, c_func, A, n_quad) - P

    # s'assurer que le signe change bien sur [0, lambda_max]
    f_max = f(lambda_max)
    while f_max > 0:      # élargir la borne sup si nécessaire
        lambda_max *= 2
        f_max = f(lambda_max)
        if lambda_max > 1e6:
            raise RuntimeError("lambda_max trop grand : vérifier le modèle.")

    return brentq(f, 1e-10, lambda_max, xtol=1e-10)


if __name__ == "__main__":
    from model import J_awgn, c_awgn

    A = 10.0
    P = A**2 / 9   # même valeur que dans le papier (§II-D, exemple numérique)

    lam_star = find_lambda_star(J_awgn, c_awgn, P, A)
    print(f"lambda* calculé numériquement : {lam_star:.6f}")

    # --- Validation contre l'approximation analytique du papier ---
    # En petit P, le papier donne : lambda* ~ 1/(2 P ln 2)   (régime gaussien)
    lam_approx = 1.0 / (2 * P * np.log(2))
    print(f"lambda* approximation papier (petit P) : {lam_approx:.6f}")
    print(f"écart relatif : {abs(lam_star - lam_approx)/lam_approx*100:.2f} %")
