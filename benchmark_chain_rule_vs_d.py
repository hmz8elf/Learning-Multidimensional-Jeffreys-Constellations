# -*- coding: utf-8 -*-
"""
benchmark_chain_rule_vs_d.py
============================
Mesure le temps de construction chain-rule pour d = 1,2,3,4,[5]
à M=4 fixé, avec les mêmes paramètres numériques que step5.py
(n_quad_marg=8, n_quad_fit=40, p=6).

Pour assurer la cohérence à toutes les dimensions (y compris impaires),
on utilise un modèle synthétique (Fisher diagonale, contrainte de puissance
quadratique) qui reflète la même structure calculatoire que le cas CSI.
"""

import sys
import time
import numpy as np
from jeffreys_baseline import BaselineConfig, JeffreysConstellationBuilder

# ── Modèle synthétique valide pour toute dimension d ──

def make_synthetic_model(d):
    """Fisher diag(1+θ_i²), c(θ) = ||θ||², identique au __main__ de jeffreys_baseline.py."""
    def J_func(theta):
        return np.diag(1.0 + theta**2)
    def c_func(theta):
        return float(np.sum(theta**2))
    return J_func, c_func

# ── Modèle CSI (pour d pair seulement, reproduit step5.py) ──

def make_csi_model_local(d):
    """Reproduit make_csi_model de validate_step4.py pour d pair."""
    nt = d // 2
    sigma2 = 0.1
    gamma_scalar = (1.0 - sigma2) / 2.0
    def J_func(theta):
        norm2 = float(np.dot(theta, theta))
        denom = 1.0 + sigma2 * norm2
        Gamma = gamma_scalar * np.eye(d)
        return (2.0 / denom) * Gamma + (4.0 * sigma2**2 / denom**2) * np.outer(theta, theta)
    def c_func(theta):
        return float(np.dot(theta, theta))
    return J_func, c_func

# ── Benchmark ──

def measure_time(d, M=4, n_quad_marg=8, n_quad_fit=40, p=6, A=10.0, model="synthetic"):
    P = A**2 / 9.0
    
    if model == "csi" and d % 2 == 0:
        J_func, c_func = make_csi_model_local(d)
    else:
        J_func, c_func = make_synthetic_model(d)
    
    config = BaselineConfig(
        A=A, d=d, M=M, p=p,
        n_quad_marg=n_quad_marg, n_quad_fit=n_quad_fit,
        domain="ball", markov_order=None
    )
    builder = JeffreysConstellationBuilder(
        J_func=J_func, c_func=c_func, P=P, config=config
    )
    
    t0 = time.perf_counter()
    builder.compute_lambda_star(verbose=False)
    X = builder.build(verbose=False)
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    print(f"  d={d} | M={M} | modèle={model} | "
          f"taille constellation={M**d} | temps = {elapsed:.4f} s")
    return elapsed

def pilot_estimate(d, M=4, n_quad_marg=8, fraction=0.1):
    """
    Estimation rapide du temps total en ne mesurant qu'une fraction
    des nœuds de quadrature (réduit n_quad_marg temporairement).
    """
    n_pilot = max(2, int(n_quad_marg * fraction))
    t_pilot = measure_time(d, M=M, n_quad_marg=n_pilot, model="synthetic")
    # Le coût scale comme n_quad_marg^(d-1) pour la marginalisation
    ratio = (n_quad_marg / n_pilot) ** max(0, d - 1)
    estimated = t_pilot * ratio
    return t_pilot, estimated

if __name__ == "__main__":
    M = 4
    results = {}
    
    print("=" * 70)
    print(f"Benchmark chain-rule: temps vs dimension d (M={M} fixé)")
    print("Modèle: synthétique (Fisher diagonale) — même algo que CSI")
    print("Paramètres: n_quad_marg=8, n_quad_fit=40, p=6, A=10")
    print("=" * 70)
    
    # ── d=1 (devrait être très rapide) ──
    print("\n--- d=1 ---")
    t = measure_time(1, M=M)
    results[1] = t
    
    # ── d=2 (référence : ~0.53s attendu) ──
    print("\n--- d=2 ---")
    t = measure_time(2, M=M)
    results[2] = t
    
    # ── d=3 ──
    print("\n--- d=3 ---")
    t = measure_time(3, M=M)
    results[3] = t
    
    # ── d=4 (référence : ~27s attendu) ──
    print("\n--- d=4 ---")
    t = measure_time(4, M=M)
    results[4] = t
    
    # ── d=5 : test pilote d'abord ──
    print("\n--- d=5 : test pilote ---")
    t_pilot, t_est = pilot_estimate(5, M=M)
    print(f"  Pilote: {t_pilot:.2f}s → estimation pleine: {t_est:.1f}s")
    
    if t_est < 25 * 60:  # < 25 minutes
        print(f"  Estimation {t_est:.0f}s < 25 min → lancement complet")
        t = measure_time(5, M=M)
        results[5] = t
    else:
        print(f"  ⚠ Estimation {t_est/60:.1f} min > 25 min → SKIP (confirmation requise)")
    
    # ── Résumé ──
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"{'d':>3} | {'M^d':>8} | {'Temps (s)':>12} | {'log₄(temps)':>12}")
    print("-" * 45)
    for d in sorted(results):
        import math
        log4t = math.log(results[d]) / math.log(4) if results[d] > 0 else float('-inf')
        print(f"{d:3d} | {M**d:8d} | {results[d]:12.4f} | {log4t:12.4f}")
    
    # ── Régression log-linéaire : log(t) = log(a) + b*d*log(M) ──
    if len(results) >= 3:
        ds = np.array(sorted(results.keys()), dtype=float)
        ts = np.array([results[int(d)] for d in ds])
        log_ts = np.log(ts)
        log_M = np.log(M)
        
        # Modèle : log(t) = c0 + b * d * log(M)
        # i.e.    log(t) = c0 + (b*log(M)) * d
        # Régression linéaire sur d
        A_mat = np.vstack([np.ones_like(ds), ds]).T
        coeffs, residuals, _, _ = np.linalg.lstsq(A_mat, log_ts, rcond=None)
        c0, slope = coeffs
        b = slope / log_M
        a = np.exp(c0)
        
        # R² 
        ss_res = np.sum((log_ts - A_mat @ coeffs)**2)
        ss_tot = np.sum((log_ts - log_ts.mean())**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        print(f"\n--- Régression log-linéaire ---")
        print(f"  Modèle : temps ≈ {a:.6f} × {M}^({b:.4f}·d)")
        print(f"  Exposant b = {b:.4f}  (théorique attendu: b ≈ 1.0)")
        print(f"  R² = {R2:.6f}")
        print(f"  Pente brute dlog(t)/dd = {slope:.4f}")
        
        if b > 1.05:
            print(f"\n  ⚠ b = {b:.4f} > 1 : les résultats suggèrent que le coût")
            print(f"    par nœud n'est pas constant avec la dimension, contrairement")
            print(f"    à l'hypothèse simplificatrice de la Proposition 6.1.")
            print(f"    Cela peut s'expliquer par la quadrature tensorielle sur les")
            print(f"    dimensions restantes dans _log_marginal(), dont le coût")
            print(f"    croît en n_quad_marg^(d-i) pour le niveau i.")
        elif b < 0.95:
            print(f"\n  ℹ b = {b:.4f} < 1 : le cache des contextes atténue")
            print(f"    la croissance par rapport à Θ(M^d) pur.")
        else:
            print(f"\n  ✓ b ≈ {b:.4f} ≈ 1 : cohérent avec Θ(M^d).")
