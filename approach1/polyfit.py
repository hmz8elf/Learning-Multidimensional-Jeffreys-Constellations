"""
Étape 4.2 : ajustement polynomial de la conditionnelle par barrière + Newton.

f_xi(theta) = sum_{k=0}^{p} xi_k * theta^k     sur [-A, A]

Minimise  L(xi) = D(f_xi || target) + gamma * D(1 || f_xi)
via la méthode de Newton, pour une suite décroissante de gamma.

xi_0 est éliminé via la contrainte de normalisation (xi_1..xi_p libres).
"""
import numpy as np


def _alpha(k: int, A: float) -> float:
    """alpha_k = integral_{-A}^{A} theta^k d theta."""
    if k % 2 == 1:
        return 0.0
    return 2 * A ** (k + 1) / (k + 1)


def xi_from_free(xi_free: np.ndarray, A: float) -> np.ndarray:
    """Reconstruit xi_0 à partir des coefficients libres xi_1..xi_p (normalisation)."""
    p = len(xi_free)
    alphas = np.array([_alpha(k, A) for k in range(1, p + 1)])
    alpha0 = _alpha(0, A)
    xi0 = (1.0 - np.dot(xi_free, alphas)) / alpha0
    return np.concatenate([[xi0], xi_free])


def eval_poly(xi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """f_xi(theta) = sum_k xi_k theta^k."""
    theta = np.atleast_1d(theta)
    powers = theta[:, None] ** np.arange(len(xi))[None, :]
    return powers @ xi


def poly_cdf(xi: np.ndarray, theta: np.ndarray, A: float) -> np.ndarray:
    """F_xi(theta) = sum_k xi_k * (theta^{k+1} - (-A)^{k+1}) / (k+1)."""
    theta = np.atleast_1d(theta)
    p = len(xi) - 1
    out = np.zeros_like(theta, dtype=float)
    for k in range(p + 1):
        out += xi[k] * (theta ** (k + 1) - (-A) ** (k + 1)) / (k + 1)
    return out


def fit_polynomial(log_target_func, degree: int, A: float,
                    gamma_schedule=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
                    n_newton_steps: int = 8, n_quad: int = 60,
                    xi_free_init: np.ndarray = None) -> np.ndarray:
    """
    Résout min_xi D(f_xi || target) + gamma*D(1||f_xi) par Newton, gamma -> 0.

    log_target_func : fonction theta -> log(target non-normalisée)(theta)
                       (le fit ne dépend que de log_target à une constante additive près,
                        car D(f_xi || target) = -H(f_xi) - E_{f_xi}[log target] + cte)
    Retourne xi (vecteur complet, taille degree+1).
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    theta_q = A * nodes
    w_q = A * weights

    log_target_q = log_target_func(theta_q)
    # normalisation de la cible en log-espace pour éviter les débordements
    log_target_q = log_target_q - np.max(log_target_q)

    if xi_free_init is None:
        xi_free = np.zeros(degree)   # point de départ : polynôme uniforme (xi_1..xi_p = 0)
    else:
        xi_free = xi_free_init.copy()

    for gamma in gamma_schedule:
        for _ in range(n_newton_steps):
            xi = xi_from_free(xi_free, A)
            f_vals = eval_poly(xi, theta_q)
            f_vals = np.clip(f_vals, 1e-12, None)   # éviter log(0)

            # psi(theta) = log f(theta) + 1 + log(target)_shift - gamma/(2A f(theta))
            # (le terme lambda*c(theta) - 0.5 log J(theta) est déjà dans log_target_q,
            #  avec un signe : on veut D(f||target) donc psi contient -log(target))
            psi = np.log(f_vals) + 1 - log_target_q - gamma / (2 * A * f_vals)

            # gradient : d f/d xi_free_j = theta^j - alpha_j/alpha_0 * theta^0  (correction xi0)
            basis = theta_q[:, None] ** np.arange(degree + 1)[None, :]   # (n_quad, degree+1)
            alphas = np.array([_alpha(k, A) for k in range(degree + 1)])
            # d xi_0 / d xi_free_j = -alpha_j / alpha_0  (j=1..degree)
            dbasis_dfree = basis[:, 1:] - np.outer(basis[:, 0], alphas[1:] / alphas[0])

            grad = (dbasis_dfree * (w_q[:, None] * psi[:, None])).sum(axis=0)

            weight_hess = 1.0 / f_vals + gamma / (2 * A * f_vals ** 2)
            hess = (dbasis_dfree.T * (w_q * weight_hess)) @ dbasis_dfree

            hess += 1e-10 * np.eye(degree)  # régularisation numérique légère
            step = np.linalg.solve(hess, grad)
            xi_free = xi_free - step

    return xi_from_free(xi_free, A)


if __name__ == "__main__":
    from model import J_awgn, c_awgn, log_w_unnormalized
    from lambda_star import find_lambda_star

    A = 10.0
    P = A**2 / 9
    lam_star = find_lambda_star(J_awgn, c_awgn, P, A)
    print(f"lambda* = {lam_star:.6f}")

    log_target = lambda th: log_w_unnormalized(th, lam_star, J_awgn, c_awgn)

    xi = fit_polynomial(log_target, degree=6, A=A)
    print("Coefficients xi* :", xi)

    # Validation : comparer f_xi à la vraie densité (Gaussienne tronquée, cas AWGN)
    theta_test = np.linspace(-A, A, 9)
    f_fit = eval_poly(xi, theta_test)

    log_true = log_target(theta_test)
    true_unnorm = np.exp(log_true - np.max(log_true))
    # normaliser true_unnorm par quadrature fine pour comparaison
    nodes, weights = np.polynomial.legendre.leggauss(400)
    th_fine = A * nodes
    log_true_fine = log_target(th_fine) - np.max(log_target(th_fine))
    Z_true = np.sum(np.exp(log_true_fine) * A * weights)
    true_norm = np.exp(log_true - np.max(log_true)) / Z_true

    print("\ntheta      f_poly(theta)   f_vraie(theta)")
    for t, fp, ft in zip(theta_test, f_fit, true_norm):
        print(f"{t:8.3f}   {fp:12.6f}   {ft:12.6f}")
