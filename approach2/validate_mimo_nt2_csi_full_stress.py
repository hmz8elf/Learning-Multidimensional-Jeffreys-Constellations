"""
validate_mimo_nt2_csi_full_stress.py
=====================================
Stress-test du RealNVP en régime fortement non-linéaire.

Objectif :
  Augmenter sigma² pour que sigma² * E[||theta||²] soit x3-x5 plus élevé
  que le test précédent (sigma²=0.1, sigma²·||theta||²≈1.1).

  Nouveau : sigma²=0.5, même P=A²/9 ≈ 11.1
  => sigma²·||theta||² ≈ 5.6  (x5 par rapport au précédent)

  Dans ce régime :
  - denom = 1 + sigma²·||theta||² ≈ 6.6
  - Le terme 2*Gamma/denom est très atténué (Gamma est réduit aussi car
    (1-sigma²)/2 = 0.25 au lieu de 0.45)
  - La correction rank-1 (theta·theta^T) domine fortement
  - La distribution cible est fortement non-gaussienne (queues plus lourdes,
    courbure du log-det J plus prononcée)

Tag : mimo_nt2_csi_full_stress (ne touche PAS aux checkpoints existants)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from config import FlowConfig, set_seed
from flow import RealNVPFlow
from train import train
from diagnostics import compute_ess
from oracle_mimo_csi_full import (
    make_anisotropic_gamma,
    solve_lambda_star_mc,
    log_w_J_mimo_full,
)


def main():
    # ------------------------------------------------------------------ #
    #  Paramètres du stress-test
    # ------------------------------------------------------------------ #
    n_t    = 2
    d      = 2 * n_t            # 4
    sigma2 = 0.5                # << augmenté de 0.1 à 0.5
    A      = 10.0
    P      = A**2 / 9.0         # ~11.11  (identique au test précédent)
    ratio  = 3.0                # anisotropie gamma, identique

    # Valeurs de référence du test précédent (sigma²=0.1)
    sigma2_prev      = 0.1
    P_prev            = P
    regime_prev       = sigma2_prev * P_prev  # ~1.11
    ess_n_prev_iso    = 0.9943
    ess_n_prev_csi    = 0.9973

    regime_new = sigma2 * P  # ~5.56

    print("=" * 65)
    print("STRESS-TEST — Régime fortement non-linéaire")
    print("=" * 65)
    print(f"n_t={n_t}, d={d}")
    print(f"sigma²          = {sigma2}   (précédent : {sigma2_prev})")
    print(f"P (puissance)   = {P:.4f}")
    print(f"sigma²·||θ||²   ≈ {regime_new:.2f}  (précédent : {regime_prev:.2f})")
    print(f"Facteur de stress : x{regime_new / regime_prev:.1f}")
    print(f"Ratio gamma     = {ratio}")
    print()

    # ------------------------------------------------------------------ #
    #  Gamma non-isotrope (même structure que le test précédent)
    # ------------------------------------------------------------------ #
    Gamma_real = make_anisotropic_gamma(n_t, sigma2, ratio=ratio)
    print(f"Gamma_real (d×d) :\n{Gamma_real.numpy()}\n")

    # Vérification : la puissance totale Tr(Gamma) change avec sigma²
    tr_gamma = torch.trace(Gamma_real).item()
    print(f"Tr(Gamma_real) = {tr_gamma:.4f}  (attendu : {d * (1 - sigma2) / 2:.4f})")
    print()

    # ------------------------------------------------------------------ #
    #  lambda* par Monte-Carlo
    # ------------------------------------------------------------------ #
    print("Résolution de lambda* (Monte-Carlo, 100k points)...")
    lambda_star = solve_lambda_star_mc(P, A, sigma2, Gamma_real, n_mc=100_000)
    print(f"lambda* = {lambda_star:.6f}\n")

    # ------------------------------------------------------------------ #
    #  Entraînement du flow
    # ------------------------------------------------------------------ #
    #  On utilise plus d'itérations (6000) et un hidden_dim=128 pour
    #  donner au flow la meilleure chance possible sur ce régime dur.
    config = FlowConfig(
        dim=d,
        n_coupling_layers=10,    # plus de couches
        hidden_dim=128,          # plus large
        lr=5e-4,
        batch_size=1024,
        n_iters=6000,            # plus d'itérations
        seed=42,
    )
    set_seed(config.seed)
    flow = RealNVPFlow(
        dim=d,
        n_coupling_layers=config.n_coupling_layers,
        hidden_dim=config.hidden_dim,
    )

    def log_w_wrapper(theta):
        return log_w_J_mimo_full(theta, lambda_star, sigma2, Gamma_real)

    print("Entraînement du flow (stress-test, tag=mimo_nt2_csi_full_stress)...")
    flow, history = train(flow, log_w_wrapper, config, tag="mimo_nt2_csi_full_stress")
    print(f"Loss finale : {history['loss'][-1]:.4f}\n")

    # ------------------------------------------------------------------ #
    #  Évaluation ESS
    # ------------------------------------------------------------------ #
    flow.eval()
    N = 5000
    with torch.no_grad():
        samples = flow.sample(N)
        log_q   = flow.log_prob(samples)
        log_w   = log_w_wrapper(samples)
        ess, ess_n = compute_ess(samples, log_w, log_q)

    # Calcul sigma²·||theta||² effectif sur les échantillons du flow
    norm2_samples = torch.sum(samples ** 2, dim=-1)
    regime_effective = sigma2 * norm2_samples.mean().item()

    print(f"ESS = {ess:.2f}, ESS/N = {ess_n:.4f}")
    print(f"sigma²·E[||θ||²] effectif = {regime_effective:.4f}")
    print()

    # ------------------------------------------------------------------ #
    #  Tableau récapitulatif
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("TABLEAU COMPARATIF  n_t=2 (d=4)")
    print(f"{'Oracle':<40} {'σ²·||θ||²':>10} {'ESS/N':>8}")
    print("-" * 60)
    print(f"{'Isotrope (σ²=0.1)':<40} {regime_prev:>10.2f} {ess_n_prev_iso:>8.4f}")
    print(f"{'CSI complet (σ²=0.1, r=3)':<40} {regime_prev:>10.2f} {ess_n_prev_csi:>8.4f}")
    print(f"{'CSI complet STRESS (σ²=0.5, r=3)':<40} {regime_effective:>10.2f} {ess_n:>8.4f}")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    #  Plots
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curve
    axes[0].plot(history["loss"])
    axes[0].set_xlabel("Itération (×100)")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss — Stress-test σ²={sigma2}")
    axes[0].grid(True)

    # Scatter θ₁ vs θ₂
    s_np = samples.detach().numpy()
    axes[1].scatter(s_np[:, 0], s_np[:, 1], s=2, alpha=0.3)
    axes[1].set_title("θ₁ vs θ₂ (stress)")
    axes[1].set_xlabel("θ₁"); axes[1].set_ylabel("θ₂")
    axes[1].grid(True); axes[1].axis("equal")

    # Distribution radiale
    norms = np.linalg.norm(s_np, axis=1)
    axes[2].hist(norms, bins=60, density=True, alpha=0.7)
    axes[2].set_title("Distribution radiale ||θ|| (stress)")
    axes[2].set_xlabel("||θ||")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("validate_mimo_nt2_csi_full_stress.png")
    print("Graphe sauvegardé : validate_mimo_nt2_csi_full_stress.png")

    return {
        "sigma2": sigma2,
        "regime_effective": regime_effective,
        "lambda_star": lambda_star,
        "ess": ess,
        "ess_n": ess_n,
        "loss_final": history["loss"][-1],
    }


if __name__ == "__main__":
    results = main()
