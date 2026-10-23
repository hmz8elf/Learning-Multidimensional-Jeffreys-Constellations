"""
amortized_mcmc.py
=================
MCMC Amorti — Proposition Indépendante via Normalizing Flow (IMHMC).

SCOPE ET LIMITATIONS (choix assumé par contrainte de temps) :
    Cette implémentation utilise le Normalizing Flow *déjà entraîné* comme
    distribution de proposition rapide et amortie. Ce n'est PAS un réseau de
    proposition entraîné spécifiquement pour le MCMC (type L2HMC / RNVMC),
    mais constitue une version légitime du concept de « MCMC amorti » au sens
    de : la proposition coûteuse à apprendre est réutilisée quasi gratuitement
    pour corriger par MH.

    Référence conceptuelle : Hoffman et al. (2019), "NeuTra-lizing Bad Geometry
    in Hamiltonian Monte Carlo" ; Parno & Marzouk (2018), transport-accelerated
    MCMC ; Gabrie et al. (2022) adaptive Monte Carlo.

ALGORITHME : Independent Metropolis-Hastings (IMH)
    1. Proposer θ' ~ q_flow  (sample du flow, O(1) passe forward)
    2. Accepter avec probabilité min(1, r) où :
           r = [w_J(θ') / q_flow(θ')] / [w_J(θ) / q_flow(θ)]
       Ce ratio est exactement le poids d'importance de SIR, donc l'acceptation
       de MH rend la chaîne géométriquement ergodique si sup_θ w_J/q_flow < ∞.

Complexité par itération : O(N_chains) passes forward du flow (trivial sur GPU).
ESS attendu ≈ min(ESS_SIR, N_chains × n_iter) selon la qualité du flow.
"""

import torch
import numpy as np
from typing import Callable, Tuple, Optional
from flow import RealNVPFlow


# ── Architecture par nt (doit correspondre aux checkpoints entraînés) ──────────
FLOW_CONFIGS = {
    2: dict(n_coupling_layers=8,  hidden_dim=64),
    4: dict(n_coupling_layers=10, hidden_dim=128),
    8: dict(n_coupling_layers=12, hidden_dim=128),
}


def load_flow(n_t: int, device: torch.device) -> RealNVPFlow:
    """Charge le flow MIMO entraîné pour une dimension donnée."""
    d = 2 * n_t
    cfg = FLOW_CONFIGS[n_t]
    flow = RealNVPFlow(dim=d, **cfg)
    ckpt_path = f"checkpoints/flow_mimo_nt{n_t}.pt"
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    flow.load_state_dict(state)
    flow.to(device)
    flow.eval()
    return flow


def _compute_log_weight(
    theta: torch.Tensor,     # (N, d)
    log_w_J_fn: Callable,    # retourne (N,) ou (N,1) — squeeze appliqué
    flow: RealNVPFlow,
) -> torch.Tensor:
    """
    Calcule le log du poids d'importance : log w_J(θ) - log q_flow(θ).
    q_flow est la densité apprise par le flow (log-prob via log-det du jacobien).
    """
    with torch.no_grad():
        log_target = log_w_J_fn(theta).view(-1)   # force shape (N,)
    log_proposal = flow.log_prob(theta).view(-1)   # force shape (N,)
    return log_target - log_proposal


def run_amortized_mh(
    log_w_J_fn: Callable,
    flow: RealNVPFlow,
    n_chains: int = 4,
    n_samples: int = 5000,
    n_warmup: int = 500,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
) -> Tuple[torch.Tensor, float, float]:
    """
    Independent Metropolis-Hastings (IMH) avec proposition = flow.

    Returns:
        samples      : (n_samples, n_chains, d) tensor
        accept_rate  : taux d'acceptation sur la phase de sampling (post-warmup)
        rhat_max     : R-hat maximal sur toutes les coordonnées (convergence)
    """
    d = flow.dim

    # Initialisation : tirer n_chains propositions du flow
    with torch.no_grad():
        theta = flow.sample(n_chains).to(device).view(n_chains, d)  # (n_chains, d)
    log_w_current = _compute_log_weight(theta, log_w_J_fn, flow)    # (n_chains,)

    def _mh_update(theta, log_w_current):
        """One IMH step: propose from flow, accept/reject."""
        with torch.no_grad():
            theta_prop = flow.sample(n_chains).to(device).view(n_chains, d)
        log_w_prop = _compute_log_weight(theta_prop, log_w_J_fn, flow)
        log_accept = log_w_prop - log_w_current
        u = torch.log(torch.rand(n_chains, device=device))
        accept = (u < log_accept)                   # (n_chains,) bool
        n_acc = accept.sum().item()
        # Update in-place on a clone to preserve gradient independence
        theta_new = theta.clone()
        theta_new[accept] = theta_prop[accept]
        log_w_new = log_w_current.clone()
        log_w_new[accept] = log_w_prop[accept]
        return theta_new, log_w_new, n_acc

    # ── Phase de warmup ──────────────────────────────────────────────────────
    for _ in range(n_warmup):
        theta, log_w_current, _ = _mh_update(theta, log_w_current)

    # ── Phase de sampling ───────────────────────────────────────────────────
    samples = torch.zeros(n_samples, n_chains, d, device=device)
    n_accepted = 0

    for i in range(n_samples):
        theta, log_w_current, n_acc = _mh_update(theta, log_w_current)
        samples[i] = theta
        n_accepted += n_acc

    accept_rate = n_accepted / (n_samples * n_chains)

    # ── Diagnostic R-hat (Gelman-Rubin) ────────────────────────────────────
    # samples : (n_samples, n_chains, d)
    # Formule classique : R-hat = sqrt( V_hat / W )
    n, m = n_samples, n_chains
    chain_means = samples.mean(dim=0)           # (m, d)
    grand_mean  = samples.mean(dim=(0, 1))      # (d,)
    B_over_n    = ((chain_means - grand_mean)**2).mean(dim=0)   # (d,)
    chain_vars  = samples.var(dim=0, unbiased=True)             # (m, d)
    W           = chain_vars.mean(dim=0)                        # (d,)
    V_hat       = ((n - 1) / n) * W + B_over_n
    rhat        = torch.sqrt(V_hat / (W + 1e-12))
    rhat_max    = rhat.max().item()

    if verbose:
        print(f"  Accept rate : {accept_rate:.3f}")
        print(f"  R-hat max   : {rhat_max:.4f}")

    return samples, accept_rate, rhat_max


def amortized_mh_sample(
    log_w_J_fn: Callable,
    flow: RealNVPFlow,
    n_samples: int = 5000,
    n_chains: int = 4,
    n_warmup: int = 500,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, float, float]:
    """
    Interface principale : retourne un flat tensor (n_samples, d) en
    combinant toutes les chaînes (après warmup, on prend 1 sample par
    itération par chaîne, puis on flatten et sous-échantillonne à n_samples).
    """
    # On lance n_samples itérations ; on garde toutes les chaînes puis
    # on aplatit et sous-échantillonne pour obtenir exactement n_samples.
    n_iter = (n_samples + n_chains - 1) // n_chains  # ceil div

    samples_all, accept_rate, rhat_max = run_amortized_mh(
        log_w_J_fn=log_w_J_fn,
        flow=flow,
        n_chains=n_chains,
        n_samples=n_iter,
        n_warmup=n_warmup,
        device=device,
    )
    # (n_iter, n_chains, d) → (n_iter * n_chains, d) → [:n_samples]
    flat = samples_all.reshape(-1, flow.dim)[:n_samples]
    return flat, accept_rate, rhat_max
