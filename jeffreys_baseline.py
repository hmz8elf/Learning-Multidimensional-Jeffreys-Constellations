# -*- coding: utf-8 -*-
"""
jeffreys_baseline.py
=====================================================================
Baseline — Construction de Constellations de Jeffreys Multidimensionnelles
Approach I : Successive Conditional Approximation
=====================================================================

Implémentation fidèle de l'Algorithme 1 du document de référence :
    "Construction de la Baseline pour les Constellations de Jeffreys
     Multidimensionnelles — Approach I"
    (Projet de stage 2026 — Jeffreys Constellations for Large-Scale MIMO)

Rappel du principe (cf. document, sections 1 à 8)
---------------------------------------------------------------------
On cherche à discrétiser le prior de Jeffreys tilté

    w_{J,lambda*}(theta) ∝ 2^{-lambda* c(theta)} sqrt(det J(theta)),   theta in [-A, A]^d

en une constellation X = {theta^(1), ..., theta^(M^d)} de taille M^d.

La méthode :
  1. Calcule lambda* par bissection pour satisfaire la contrainte de
     puissance moyenne E[c(theta)] <= P (section 3).
  2. Désintègre la loi cible en produit de conditionnelles
     w(theta) = prod_i kappa_i(theta_i | theta_1:i-1)  (section 2).
  3. Approxime chaque conditionnelle kappa_i par un polynôme f_xi de
     degré p, ajusté par minimisation de la divergence de
     Kullback-Leibler sous contrainte de positivité (barrière
     logarithmique) et de normalisation, résolue par la méthode de
     Newton (section 4).
  4. Échantillonne chaque f_xi sur une grille déterministe régulière
     S_M par inversion de sa CDF polynomiale fermée (section 4.6).
  5. Construit ainsi récursivement un arbre de profondeur d, dont les
     feuilles forment la constellation brute, puis applique une mise
     à l'échelle finale pour respecter exactement la contrainte de
     puissance (étape 6 de l'algorithme).

Complexité (section 6) : Θ(M^d) en général, réductible à O(d * M^D)
sous une structure markovienne d'ordre D — d'où le paramètre optionnel
`markov_order` de `BaselineConfig`.

Ce module est un outil de recherche / prototypage : les intégrales
multidimensionnelles (quadrature de Gauss-Legendre tensorielle) ont un
coût qui croît vite avec la dimension restante à intégrer ; il est
prévu pour des dimensions d modestes (typiquement d <= 4) et des
ordres de quadrature raisonnables, conformément à l'esprit "baseline"
du document (une méthode de référence, pas nécessairement la plus
scalable — cf. Approach II évoquée en conclusion).
---------------------------------------------------------------------
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss


# =====================================================================
# 0. Utilitaires de quadrature
# =====================================================================

def gauss_legendre_nodes_weights(a: float, b: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nœuds et poids de quadrature de Gauss-Legendre, redimensionnés de
    [-1, 1] vers [a, b].

    Utilisé pour toutes les intégrales sur des intervalles bornés du
    document (marginalisation section 4.1, ajustement polynomial
    section 4.4-4.5).
    """
    x, w = leggauss(n)
    x_scaled = 0.5 * (b - a) * x + 0.5 * (b + a)
    w_scaled = 0.5 * (b - a) * w
    return x_scaled, w_scaled


# =====================================================================
# 1. Configuration de l'algorithme (Table 2 du document)
# =====================================================================

@dataclass
class BaselineConfig:
    """Paramètres fixés avant tout calcul (Table 2 du document)."""

    A: float                      # domaine Theta = [-A, A]^d
    d: int                        # dimension du paramètre
    M: int                        # nombre de points souhaités par coordonnée
    p: int                        # degré du polynôme d'approximation

    # Ordres de quadrature (compromis précision / coût, non fixés par
    # le document — laissés au choix de l'implémentation) :
    n_quad_marg: int = 8          # quadrature pour la marginalisation (4.1)
    n_quad_fit: int = 40          # quadrature pour l'ajustement polynomial (4.5)

    # Bissection sur lambda (section 3) :
    lambda_max: float = 50.0
    lambda_tol: float = 1e-8

    # Suite décroissante {gamma_k} de la méthode de barrière (4.4) :
    gamma_sequence: Sequence[float] = field(
        default_factory=lambda: (1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3,
                                  1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)
    )

    # Newton amorti (damped Newton) pour chaque valeur de gamma :
    newton_max_iter: int = 60
    newton_tol: float = 1e-10

    # Bissection pour l'inversion de la CDF (4.6) :
    bisection_cdf_tol: float = 1e-12
    bisection_cdf_max_iter: int = 100

    # Réduction markovienne optionnelle (Proposition 6.2) : si fixé à
    # D, seules les D dernières coordonnées du préfixe sont utilisées
    # comme contexte, avec mise en cache -> coût O(d * M^D) au lieu de
    # Θ(M^d).
    markov_order: Optional[int] = None


# =====================================================================
# 2. Constructeur principal
# =====================================================================

class JeffreysConstellationBuilder:
    """
    Construit une constellation approchant w_{J,lambda*} selon
    l'Algorithme 1 du document (Successive Conditional Approximation).

    Paramètres
    ----------
    J_func : Callable[[np.ndarray], np.ndarray]
        theta (vecteur de taille d) -> matrice de Fisher J(theta) (d x d).
    c_func : Callable[[np.ndarray], float]
        theta -> coût scalaire c(theta) (contrainte de puissance).
    P : float
        Contrainte de puissance moyenne E[c(theta)] <= P.
    config : BaselineConfig
        Paramètres numériques (voir Table 2 du document).
    """

    def __init__(self, J_func: Callable[[np.ndarray], np.ndarray],
                 c_func: Callable[[np.ndarray], float],
                 P: float,
                 config: BaselineConfig):
        self.J = J_func
        self.c = c_func
        self.P = P
        self.cfg = config
        self.ln2 = np.log(2.0)
        self.lambda_star: Optional[float] = None

        # Nœuds/poids de quadrature réutilisés (évite de les
        # recalculer à chaque appel) :
        self._marg_nodes, self._marg_weights = gauss_legendre_nodes_weights(
            -config.A, config.A, config.n_quad_marg)
        self._fit_nodes, self._fit_weights = gauss_legendre_nodes_weights(
            -config.A, config.A, config.n_quad_fit)

        # alpha_k = integral_{-A}^{A} theta^k dtheta, k = 0..p (eq. 4.3)
        self._alpha = np.array([self._alpha_k(k) for k in range(config.p + 1)])

    # -----------------------------------------------------------
    # Densité non-normalisée g(theta) = 2^{-lambda* c(theta)} sqrt(det J(theta))
    # -----------------------------------------------------------
    def _alpha_k(self, k: int) -> float:
        """alpha_k = ∫_{-A}^{A} theta^k dtheta (eq. 13)."""
        A = self.cfg.A
        return (A ** (k + 1) - (-A) ** (k + 1)) / (k + 1)

    def log_g(self, theta: np.ndarray) -> float:
        """
        log g(theta) = -lambda* ln2 c(theta) + 0.5 log det J(theta)

        (log de la densité (1) non normalisée, en espace log pour la
        stabilité numérique — cf. "4.1 : ... en espace logarithmique"
        dans le document).
        """
        Jm = np.asarray(self.J(theta), dtype=float)
        sign, logdet = np.linalg.slogdet(Jm)
        if sign <= 0:
            # det J(theta) <= 0 : configuration dégénérée, densité nulle
            logdet = -1e6
        return -self.lambda_star * self.c(theta) * self.ln2 + 0.5 * logdet

    # -----------------------------------------------------------
    # Étape 1 — Calcul de lambda* (section 3)
    # -----------------------------------------------------------
    def _M_of_lambda(self, lam: float) -> float:
        """
        M(lambda) = E_{theta ~ w_{J,lambda}}[c(theta)]  (Définition 3.1),
        calculé par quadrature de Gauss-Legendre tensorielle complète
        sur Theta = [-A, A]^d.
        """
        cfg = self.cfg
        nodes, weights = self._marg_nodes, self._marg_weights

        logs = []
        cs = []
        ws = []
        for idx in itertools.product(range(len(nodes)), repeat=cfg.d):
            theta = np.array([nodes[k] for k in idx])
            w = np.prod([weights[k] for k in idx])
            Jm = np.asarray(self.J(theta), dtype=float)
            sign, logdet = np.linalg.slogdet(Jm)
            if sign <= 0:
                logdet = -1e6
            log_val = -lam * self.c(theta) * self.ln2 + 0.5 * logdet
            logs.append(log_val)
            cs.append(self.c(theta))
            ws.append(w)

        logs = np.array(logs)
        cs = np.array(cs)
        ws = np.array(ws)
        m = logs.max()
        unnorm = ws * np.exp(logs - m)          # proportionnel à g(theta) * poids
        den = unnorm.sum()
        num = (unnorm * cs).sum()
        return num / den if den > 0 else 0.0

    def compute_lambda_star(self, verbose: bool = False) -> float:
        """
        Résout M(lambda*) = P par bissection (Corollaire 3.3, garantie
        de convergence par la stricte monotonie du Lemme 3.2).
        """
        cfg = self.cfg
        m0 = self._M_of_lambda(0.0)
        if m0 <= self.P:
            self.lambda_star = 0.0
            if verbose:
                print(f"M(0) = {m0:.6g} <= P = {self.P:.6g} -> lambda* = 0 "
                      f"(contrainte inactive)")
            return 0.0

        lo, hi = 0.0, cfg.lambda_max
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            self.lambda_star = mid  # nécessaire car _M_of_lambda dépend implicitement
            val = self._M_of_lambda(mid)
            if val > self.P:
                lo = mid
            else:
                hi = mid
            if hi - lo < cfg.lambda_tol:
                break
        self.lambda_star = 0.5 * (lo + hi)
        if verbose:
            print(f"lambda* = {self.lambda_star:.6g} "
                  f"(M(lambda*) = {self._M_of_lambda(self.lambda_star):.6g}, P = {self.P:.6g})")
        return self.lambda_star

    # -----------------------------------------------------------
    # Étape 4.1 — Marginale log w_i(theta_1:i-1, theta_i) (section 2.1)
    # -----------------------------------------------------------
    def _log_marginal(self, prefix: Tuple[float, ...], theta_i: float) -> float:
        """
        log w_i(theta_1:i-1, theta_i)
            = log ∫_{Theta_{i+1:d}} g(theta_1:i-1, theta_i, vartheta_{i+1:d}) dvartheta_{i+1:d}

        Calculé par quadrature de Gauss-Legendre tensorielle sur les
        (d - i) dimensions restantes (section 4.1). Si i = d (dernier
        niveau), aucune intégration n'est nécessaire.

        Note : cette quantité n'est pas normalisée en theta_i (elle
        diffère de kappa_i(theta_i | theta_1:i-1) par un facteur
        multiplicatif constant w_{i-1}(theta_1:i-1) indépendant de
        theta_i). Comme l'ajustement polynomial (section 4.2) contraint
        f_xi à être normalisée (∫f_xi = 1), ce facteur constant
        n'affecte pas l'argmin xi* — cf. docstring de `_fit_polynomial`.
        """
        d = self.cfg.d
        n_remaining = d - len(prefix) - 1

        if n_remaining == 0:
            theta_full = np.array(list(prefix) + [theta_i])
            return self.log_g(theta_full)

        nodes, weights = self._marg_nodes, self._marg_weights
        logs = []
        ws = []
        for idx in itertools.product(range(len(nodes)), repeat=n_remaining):
            rest = [nodes[k] for k in idx]
            w = np.prod([weights[k] for k in idx])
            theta_full = np.array(list(prefix) + [theta_i] + rest)
            logs.append(self.log_g(theta_full))
            ws.append(w)
        logs = np.array(logs)
        ws = np.array(ws)
        m = logs.max()
        return m + np.log(np.sum(ws * np.exp(logs - m)))

    # -----------------------------------------------------------
    # Étape 4.2 — Ajustement polynomial par barrière + Newton (section 4)
    # -----------------------------------------------------------
    def _fit_polynomial(self, log_shape_at_fit_nodes: np.ndarray) -> np.ndarray:
        """
        Résout (15) : min_xi  D(f_xi || kappa_i) + gamma * D(1 || f_xi)

        où f_xi(theta) = sum_{k=0}^p xi_k theta^k, la contrainte de
        normalisation ∫f_xi = 1 étant éliminée via (13) :

            xi_0 = (1 - sum_{k=1}^p xi_k alpha_k) / alpha_0

        Les variables libres sont xi_1..xi_p. On résout par continuation
        sur une suite décroissante {gamma_k} (méthode de barrière, 4.4),
        chaque sous-problème étant résolu par une méthode de Newton
        amortie (line search par backtracking pour garantir f_xi > 0
        sur les nœuds de quadrature à chaque itération), avec réamorçage
        (warm start) depuis la solution du gamma précédent.

        `log_shape_at_fit_nodes` : valeurs de log w_i(theta_1:i-1, .)
        (voir `_log_marginal`) évaluées aux nœuds de quadrature
        `self._fit_nodes`. Comme expliqué dans `_log_marginal`, utiliser
        cette quantité non normalisée à la place de log kappa_i ne
        change L(xi) que d'une constante additive indépendante de xi
        (car ∫f_xi = 1 est imposé), donc n'affecte pas xi*.

        Gradient et Hessienne (dérivés directement, cf. section 4.5) :
        en posant phi_k(theta) := theta^k - alpha_k/alpha_0 (k=1..p),
        de sorte que d f_xi/d xi_k = phi_k(theta) [car d xi_0/d xi_k
        = -alpha_k/alpha_0] :

            dL/dxi_k = ∫ phi_k(theta) [log f_xi(theta) + 1 - logkappa(theta)] dtheta
                       - (gamma / 2A) ∫ phi_k(theta) / f_xi(theta) dtheta

            d2L/dxi_k dxi_l = ∫ phi_k phi_l [1/f_xi + (gamma/2A)/f_xi^2] dtheta

        La Hessienne est définie positive (Proposition 4.2, vecteurs de
        Vandermonde linéairement indépendants), donc L est strictement
        convexe : Newton converge quadratiquement une fois dans le
        bassin d'attraction, la line search assurant la faisabilité
        globale (f_xi > 0) à chaque pas.
        """
        cfg = self.cfg
        p = cfg.p
        A = cfg.A
        nodes, weights = self._fit_nodes, self._fit_weights
        alpha = self._alpha

        # theta_pows[k, :] = nodes ** k, k = 0..p
        theta_pows = np.vstack([nodes ** k for k in range(p + 1)])

        # phi_k(theta) = theta^k - alpha_k/alpha_0, k = 1..p  (shape p x n_quad_fit)
        phi = theta_pows[1:] - (alpha[1:, None] / alpha[0]) * theta_pows[0]

        def f_of(xi_free: np.ndarray) -> np.ndarray:
            xi0 = (1.0 - np.dot(xi_free, alpha[1:])) / alpha[0]
            return xi0 + xi_free @ theta_pows[1:]

        xi_free = np.zeros(p)  # initialisation : densité uniforme (f_xi = 1/2A)

        for gamma in cfg.gamma_sequence:
            for _ in range(cfg.newton_max_iter):
                fvals = f_of(xi_free)
                fvals_safe = np.maximum(fvals, 1e-12)
                logf = np.log(fvals_safe)

                bracket = logf + 1.0 - log_shape_at_fit_nodes
                grad = (phi * (weights * bracket)[None, :]).sum(axis=1)
                grad -= (gamma / (2 * A)) * (phi * (weights / fvals_safe)[None, :]).sum(axis=1)

                curvature = weights * (1.0 / fvals_safe + (gamma / (2 * A)) / fvals_safe ** 2)
                H = (phi * curvature[None, :]) @ phi.T

                try:
                    step = np.linalg.solve(H, grad)
                except np.linalg.LinAlgError:
                    step = np.linalg.lstsq(H, grad, rcond=None)[0]

                # Backtracking line search : garantit f_xi > 0 sur tous
                # les nœuds de quadrature après le pas.
                t = 1.0
                xi_trial = xi_free
                for _ in range(50):
                    xi_trial = xi_free - t * step
                    if np.all(f_of(xi_trial) > 1e-10):
                        break
                    t *= 0.5
                xi_free = xi_trial

                if np.linalg.norm(step) * t < cfg.newton_tol:
                    break

        xi0 = (1.0 - np.dot(xi_free, alpha[1:])) / alpha[0]
        return np.concatenate([[xi0], xi_free])

    # -----------------------------------------------------------
    # Vérification Étape 1 — positivité de f_xi hors des nœuds de fit
    # -----------------------------------------------------------
    def _check_positivity(self, xi: np.ndarray, n_check: int = 500) -> float:
        """
        La barrière logarithmique (section 4.4) ne garantit f_xi > 0
        qu'aux `n_quad_fit` nœuds de quadrature utilisés pendant
        l'optimisation (cf. `_fit_polynomial`). Comme f_xi est un
        polynôme, rien n'empêche *a priori* une violation entre deux
        nœuds consécutifs (oscillation de Runge, degré p élevé).

        On revérifie ici sur une grille fine indépendante (n_check
        points, uniforme sur [-A, A]) et on retourne le minimum de
        f_xi observé. Une valeur négative ou proche de zéro signale
        que le degré p est trop élevé pour n_quad_fit, ou que gamma
        n'a pas assez décru.
        """
        A = self.cfg.A
        p = len(xi) - 1
        grid = np.linspace(-A, A, n_check)
        powers = np.vstack([grid ** k for k in range(p + 1)])
        f_vals = xi @ powers
        min_val = float(f_vals.min())
        if min_val <= 1e-8:
            import warnings
            warnings.warn(
                f"f_xi non strictement positive sur la grille fine de "
                f"vérification (min = {min_val:.3e}). Augmenter n_quad_fit "
                f"ou allonger gamma_sequence.", stacklevel=2)
        return min_val

    # -----------------------------------------------------------
    # Étape 4.3 — Inversion de la CDF polynomiale (section 4.6, eq. 19)
    # -----------------------------------------------------------
    def _invert_cdf(self, xi: np.ndarray, u: float) -> float:
        """
        Résout F_xi(theta) = u par bissection, où

            F_xi(theta) = sum_{k=0}^p xi_k [theta^{k+1} - (-A)^{k+1}] / (k+1)

        est la CDF fermée de f_xi (eq. 19). F_xi est strictement
        croissante tant que f_xi > 0 sur [-A, A] (garanti par la
        barrière logarithmique de l'ajustement précédent).
        """
        A = self.cfg.A
        p = len(xi) - 1

        def F(theta: float) -> float:
            s = 0.0
            for k in range(p + 1):
                s += xi[k] * (theta ** (k + 1) - (-A) ** (k + 1)) / (k + 1)
            return s

        lo, hi = -A, A
        for _ in range(self.cfg.bisection_cdf_max_iter):
            mid = 0.5 * (lo + hi)
            if F(mid) < u:
                lo = mid
            else:
                hi = mid
            if hi - lo < self.cfg.bisection_cdf_tol:
                break
        return 0.5 * (lo + hi)

    # -----------------------------------------------------------
    # Clé de contexte (réduction markovienne optionnelle, Prop. 6.2)
    # -----------------------------------------------------------
    def _context_key(self, level: int, prefix: Tuple[float, ...]) -> Tuple[int, Tuple[float, ...]]:
        """
        Si `markov_order` = D est fixé, seules les D dernières
        coordonnées du préfixe déterminent le contexte utilisé pour
        l'ajustement polynomial, ce qui permet une mise en cache
        exacte (les valeurs de la grille S_M étant réutilisées de
        façon identique à chaque nœud, cf. section 2.3) et réduit le
        coût par niveau de M^{i-1} à M^D (Proposition 6.2).

        IMPORTANT : le niveau `level` fait partie de la clé. Sans ça,
        deux niveaux différents partageraient le même ξ* dès que
        markov_order tronque au même contexte (ex. D=0 -> contexte
        vide à tous les niveaux), ce qui n'est valide QUE si les
        coordonnées sont exchangeables (même loi marginale conditionnelle
        à chaque position i). Ce n'est pas garanti en général (ex. cas
        III-D du papier, où la position dans le vecteur peut compter) ;
        inclure `level` rend le cache correct par construction, quitte
        à perdre le partage inter-niveaux dans les cas exchangeables
        (partage qu'on peut réactiver explicitement si justifié, mais
        pas par défaut).
        """
        if self.cfg.markov_order is not None:
            trunc = prefix[-self.cfg.markov_order:] if self.cfg.markov_order > 0 else ()
        else:
            trunc = prefix
        return (level, tuple(np.round(trunc, 12)))

    # -----------------------------------------------------------
    # Algorithme complet (Algorithm 1 / section 8)
    # -----------------------------------------------------------
    def build(self, verbose: bool = True) -> np.ndarray:
        """
        Exécute l'Algorithme 1 en entier et retourne la constellation
        finale X_final, de forme (M^d, d).
        """
        cfg = self.cfg

        # --- Étape 1 : lambda* ---
        self.compute_lambda_star(verbose=verbose)

        # --- Étape 2 : grille d'échantillonnage S_M (eq. 7) ---
        k = np.arange(1, cfg.M + 1)
        S_M = (k - 0.5) / cfg.M

        # --- Étape 3 : initialisation de l'arbre L_0 = {emptyset} ---
        L: List[Tuple[float, ...]] = [()]

        cache: dict = {}

        # --- Étape 4 : boucle sur les dimensions ---
        for i in range(1, cfg.d + 1):
            L_next: List[Tuple[float, ...]] = []
            for prefix in L:
                key = self._context_key(i, prefix)
                if key in cache:
                    xi_star = cache[key]
                else:
                    # 4.1 : marginale par quadrature de Gauss-Legendre
                    log_shape = np.array([
                        self._log_marginal(prefix, theta_i)
                        for theta_i in self._fit_nodes
                    ])
                    # 4.2 : ajustement polynomial (barrière + Newton)
                    xi_star = self._fit_polynomial(log_shape)
                    # Vérification Étape 1 : positivité hors nœuds de fit
                    self._check_positivity(xi_star)
                    cache[key] = xi_star

                # 4.3 : inversion de la CDF pour chaque u de la grille
                for u in S_M:
                    theta_i_hat = self._invert_cdf(xi_star, u)
                    # 4.4 : extension de l'arbre
                    L_next.append(prefix + (theta_i_hat,))
            L = L_next
            if verbose:
                print(f"Niveau {i}/{cfg.d} terminé : {len(L)} nœuds "
                      f"({len(cache)} contextes distincts mis en cache)")

        # --- Étape 5 : terminaison ---
        X = np.array(L)  # shape (M^d, d)

        # --- Étape 6 : mise à l'échelle de puissance ---
        c_vals = np.array([self.c(theta) for theta in X])
        mean_c = c_vals.mean()
        c_P = min(1.0, np.sqrt(self.P / mean_c)) if mean_c > 0 else 1.0
        X_final = c_P * X

        if verbose:
            print(f"Mise à l'échelle : c_P = {c_P:.6g}, "
                  f"puissance moyenne finale = {(np.array([self.c(t) for t in X_final])).mean():.6g} "
                  f"(cible P = {self.P:.6g})")

        self.cache = cache  # exposé pour inspection (vérification Étape 1)
        return X_final


# =====================================================================
# 3. Exemple d'utilisation (canal MIMO jouet, d = 2)
# =====================================================================

if __name__ == "__main__":
    # Exemple synthétique : theta = (theta_1, theta_2, theta_3, theta_4, theta_5) in [-A, A]^5.
    # J(theta) = diag(1 + theta_1^2, 1 + theta_2^2, 1 + theta_3^2, 1 + theta_4^2, 1 + theta_5^2)  (matrice de Fisher jouet)
    # c(theta) = ||theta||^2                          (contrainte de puissance)

    def J_func(theta: np.ndarray) -> np.ndarray:
        return np.diag(1.0 + theta ** 2)

    def c_func(theta: np.ndarray) -> float:
        return float(np.sum(theta ** 2))

    config = BaselineConfig(
        A=3.0,
        d=3,
        M=10,
        p=5,
        n_quad_marg=8,
        n_quad_fit=40,
    )

    builder = JeffreysConstellationBuilder(J_func=J_func, c_func=c_func, P=1.5, config=config)
    X_final = builder.build(verbose=True)

    print("\nConstellation finale (5 premiers points) :")
    print(X_final)
    print(f"\nTaille de la constellation : {X_final.shape[0]} points en dimension {X_final.shape[1]}")