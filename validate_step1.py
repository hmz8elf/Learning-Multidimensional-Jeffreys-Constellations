# -*- coding: utf-8 -*-
"""
validate_step1.py
=====================================================================
Étape 1 — Vérification du pipeline sur le cas jouet actuel
=====================================================================

Trois vérifications, comme convenu :

  A. Positivité de f_xi sur une grille fine (pas seulement aux nœuds
     de quadrature utilisés pour l'ajustement).
  B. Cohérence inter-contextes : dans l'exemple jouet actuel
     (J = diag(1+theta_i^2), c = sum theta_i^2), le problème est
     séparable ET les coordonnées sont exchangeables, donc tous les
     xi* mis en cache à un même niveau i devraient être quasi
     identiques, quel que soit le préfixe.
  B'. Idem À TRAVERS les niveaux (car exchangeable ici) : xi* au
     niveau 1, 2, 3... devrait aussi être quasi identique.
  C. markov_order=0 doit donner la même constellation que
     markov_order=None (aux erreurs de quadrature près), mais
     beaucoup plus vite -- avec la clé de cache maintenant correcte
     (niveau + contexte tronqué), voir jeffreys_baseline.py.
"""

import time
import numpy as np

from jeffreys_baseline import BaselineConfig, JeffreysConstellationBuilder


def J_func(theta: np.ndarray) -> np.ndarray:
    return np.diag(1.0 + theta ** 2)


def c_func(theta: np.ndarray) -> float:
    return float(np.sum(theta ** 2))


def run(markov_order, verbose=False):
    config = BaselineConfig(
        A=3.0, d=3, M=10, p=5,
        n_quad_marg=8, n_quad_fit=40,
        markov_order=markov_order,
    )
    builder = JeffreysConstellationBuilder(J_func=J_func, c_func=c_func, P=1.5, config=config)
    t0 = time.perf_counter()
    X = builder.build(verbose=verbose)
    dt = time.perf_counter() - t0
    return builder, X, dt


def main():
    print("=" * 70)
    print("A. Baseline complète (markov_order=None) + positivité + cohérence")
    print("=" * 70)
    builder_full, X_full, dt_full = run(markov_order=None, verbose=True)
    print(f"\nTemps (markov_order=None) : {dt_full:.3f} s")

    # --- B. Cohérence des xi* à un même niveau ---
    print("\n" + "=" * 70)
    print("B. Cohérence des xi* mis en cache (séparabilité de l'exemple jouet)")
    print("=" * 70)
    by_level = {}
    for (level, ctx), xi in builder_full.cache.items():
        by_level.setdefault(level, []).append(xi)

    for level, xi_list in sorted(by_level.items()):
        xis = np.array(xi_list)
        if len(xis) > 1:
            ref = xis[0]
            max_dev = np.max(np.abs(xis - ref))
        else:
            max_dev = 0.0
        print(f"Niveau {level} : {len(xi_list)} contextes distincts, "
              f"écart max entre xi* = {max_dev:.3e} "
              f"{'(OK, séparabilité confirmée)' if max_dev < 1e-6 else '(!) écart notable'}")

    # --- B'. Cohérence à travers les niveaux (exchangeabilité) ---
    print("\nCohérence À TRAVERS les niveaux (coordonnées exchangeables ici) :")
    all_xi = [xi for xi_list in by_level.values() for xi in xi_list]
    all_xi = np.array(all_xi)
    ref = all_xi[0]
    max_dev_cross = np.max(np.abs(all_xi - ref))
    print(f"Écart max tous niveaux confondus = {max_dev_cross:.3e} "
          f"{'(OK)' if max_dev_cross < 1e-6 else '(!) attendu si dims exchangeables)'}")

    # --- C. markov_order=0 : même résultat, plus rapide ---
    print("\n" + "=" * 70)
    print("C. markov_order=0 : correction et gain de vitesse")
    print("=" * 70)
    builder_mk0, X_mk0, dt_mk0 = run(markov_order=0, verbose=True)
    print(f"\nTemps (markov_order=0)   : {dt_mk0:.3f} s")
    print(f"Temps (markov_order=None): {dt_full:.3f} s")
    print(f"Accélération             : x{dt_full / dt_mk0:.1f}")

    same = np.allclose(X_full, X_mk0, atol=1e-9)
    max_diff = np.max(np.abs(X_full - X_mk0))
    print(f"\nConstellations identiques (atol=1e-9) : {same}")
    print(f"Écart max entre les deux constellations : {max_diff:.3e}")

    print(f"\nNombre de contextes en cache : "
          f"None -> {len(builder_full.cache)}, "
          f"markov_order=0 -> {len(builder_mk0.cache)} (devrait être = d = {builder_mk0.cfg.d})")

    # --- Diagnostic : quels contextes violent la positivité, et de combien ? ---
    print("\n" + "=" * 70)
    print("Diagnostic positivité : localisation précise des violations")
    print("=" * 70)
    n_check = 2000
    A = builder_full.cfg.A
    grid = np.linspace(-A, A, n_check)
    worst = []
    for (level, ctx), xi in builder_full.cache.items():
        p = len(xi) - 1
        powers = np.vstack([grid ** k for k in range(p + 1)])
        f_vals = xi @ powers
        min_val = float(f_vals.min())
        worst.append((min_val, level, ctx))
    worst.sort(key=lambda t: t[0])
    n_violations = sum(1 for m, _, _ in worst if m <= 1e-8)
    print(f"Contextes en violation (f_xi <= 0 quelque part) : "
          f"{n_violations} / {len(worst)}")
    print("5 pires cas :")
    for min_val, level, ctx in worst[:5]:
        print(f"  niveau={level}, contexte(tronqué)={ctx}, min f_xi = {min_val:.3e}")


if __name__ == "__main__":
    main()
