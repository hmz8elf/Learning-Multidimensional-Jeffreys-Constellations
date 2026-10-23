"""
validate_mimo_nt2_csi_full.py
==============================
Entraîne et évalue un RealNVP sur le cas MIMO CSI imparfait COMPLET
(Gamma non-isotrope, Section III-D) pour n_t=2, d=4.
Compare avec le résultat isotrope déjà obtenu.
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
    n_t   = 2
    d     = 2 * n_t
    sigma2 = 0.1
    A     = 10.0
    P     = A**2 / 9.0
    ratio = 3.0          # anisotropie : eigenvalue max / min = 3

    print("=" * 60)
    print(f"MIMO CSI Imparfait COMPLET  n_t={n_t}, d={d}")
    print(f"sigma2={sigma2}, ratio_gamma={ratio}")
    print("=" * 60)

    # --- Gamma non-isotrope ---
    Gamma_real = make_anisotropic_gamma(n_t, sigma2, ratio=ratio)
    print(f"Gamma_real (d x d) :\n{Gamma_real.numpy()}\n")

    # --- lambda* par MC ---
    print("Résolution de lambda* (Monte-Carlo)...")
    lambda_star = solve_lambda_star_mc(P, A, sigma2, Gamma_real, n_mc=100_000)
    print(f"lambda* = {lambda_star:.6f}\n")

    # --- Config flow ---
    config = FlowConfig(
        dim=d,
        n_coupling_layers=8,
        hidden_dim=64,
        lr=5e-4,
        batch_size=1024,
        n_iters=4000,
        seed=42,
    )
    set_seed(config.seed)
    flow = RealNVPFlow(dim=d,
                       n_coupling_layers=config.n_coupling_layers,
                       hidden_dim=config.hidden_dim)

    def log_w_wrapper(theta):
        return log_w_J_mimo_full(theta, lambda_star, sigma2, Gamma_real)

    # --- Entraînement ---
    print("Entraînement du flow (CSI full)...")
    flow, history = train(flow, log_w_wrapper, config, tag="mimo_nt2_csi_full")
    print(f"Loss finale : {history['loss'][-1]:.4f}\n")

    # --- ESS ---
    flow.eval()
    N = 5000
    with torch.no_grad():
        samples = flow.sample(N)
        log_q   = flow.log_prob(samples)
        log_w   = log_w_wrapper(samples)
        ess, ess_n = compute_ess(samples, log_w, log_q)

    print(f"ESS = {ess:.2f}, ESS/N = {ess_n:.4f}\n")

    # --- Résultats référence isotrope ---
    ess_n_isotrope = 0.9943   # valeur déjà obtenue (validate_mimo_nt2.py)

    print("=" * 60)
    print("TABLEAU COMPARATIF  n_t=2 (d=4)")
    print(f"{'Oracle':<35} {'ESS/N':>8}")
    print("-" * 45)
    print(f"{'Isotrope (Gamma = gamma*I)':<35} {ess_n_isotrope:>8.4f}")
    print(f"{'CSI complet (Gamma anisotrope)':<35} {ess_n:>8.4f}")
    print("=" * 60)

    # --- Plot comparaison : loss curve ---
    plt.figure(figsize=(8, 4))
    plt.plot(history["loss"], label="CSI complet (Gamma anisotrope)")
    plt.xlabel("Itération (x100)")
    plt.ylabel("Loss")
    plt.title("Loss – MIMO CSI complet vs Isotrope (n_t=2)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("validate_mimo_nt2_csi_full_loss.png")
    print("Graphe loss sauvegardé : validate_mimo_nt2_csi_full_loss.png")

    # --- Plot scatter dim 0 vs dim 1 ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    samples_np = samples.detach().numpy()

    # Load isotrope samples si disponibles, sinon juste montrer csi_full
    axes[0].scatter(samples_np[:, 0], samples_np[:, 1], s=2, alpha=0.3)
    axes[0].set_title("CSI complet (θ₁ vs θ₂)")
    axes[0].set_xlabel("θ₁"); axes[0].set_ylabel("θ₂")
    axes[0].grid(True); axes[0].axis("equal")

    # Marginal histograms
    axes[1].hist(np.linalg.norm(samples_np, axis=1), bins=60, density=True,
                 alpha=0.7, label="CSI complet")
    axes[1].set_title("Distribution radiale ||θ||")
    axes[1].set_xlabel("||θ||"); axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("validate_mimo_nt2_csi_full_scatter.png")
    print("Graphe scatter sauvegardé : validate_mimo_nt2_csi_full_scatter.png")

    return ess_n, ess_n_isotrope

if __name__ == "__main__":
    main()
