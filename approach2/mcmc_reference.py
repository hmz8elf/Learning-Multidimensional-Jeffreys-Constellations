import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Callable, Tuple

def leapfrog(q, p, step_size, log_prob_fn, grad_log_prob_fn):
    p = p + 0.5 * step_size * grad_log_prob_fn(q)
    q = q + step_size * p
    p = p + 0.5 * step_size * grad_log_prob_fn(q)
    return q, p

def run_hmc(
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    n_chains: int = 4,
    n_samples: int = 5000,
    n_warmup: int = 1000,
    step_size: float = 0.1,
    n_leapfrog_steps: int = 10,
    init_q: torch.Tensor = None,
    device: str = "cpu"
) -> Tuple[torch.Tensor, float]:
    """
    Runs Hamiltonian Monte Carlo.
    q: (n_chains, dim)
    Returns:
        samples: (n_samples, n_chains, dim)
        accept_rate: float
    """
    if init_q is None:
        q = torch.randn(n_chains, dim, device=device)
    else:
        q = init_q.clone().to(device)

    def compute_grad(x):
        x = x.clone().requires_grad_(True)
        log_prob = log_prob_fn(x)
        grad = torch.autograd.grad(log_prob.sum(), x)[0]
        return grad

    samples = torch.zeros(n_samples, n_chains, dim, device=device)
    accepts = torch.zeros(n_chains, device=device)

    # Warmup
    current_log_prob = log_prob_fn(q)
    for w in range(n_warmup):
        p = torch.randn_like(q)
        current_K = 0.5 * torch.sum(p**2, dim=-1)
        
        q_new, p_new = q.clone(), p.clone()
        for _ in range(n_leapfrog_steps):
            q_new, p_new = leapfrog(q_new, p_new, step_size, log_prob_fn, compute_grad)
        
        new_log_prob = log_prob_fn(q_new)
        new_K = 0.5 * torch.sum(p_new**2, dim=-1)
        
        # Metropolis acceptance
        log_accept_ratio = new_log_prob - current_log_prob + current_K - new_K
        u = torch.log(torch.rand(n_chains, device=device))
        accept = u < log_accept_ratio
        
        q[accept] = q_new[accept]
        current_log_prob[accept] = new_log_prob[accept]
        
        # Adapt step size
        accept_rate_w = accept.float().mean().item()
        if accept_rate_w < 0.65:
            step_size *= 0.99
        elif accept_rate_w > 0.8:
            step_size *= 1.01

    # Sampling
    for i in range(n_samples):
        p = torch.randn_like(q)
        current_K = 0.5 * torch.sum(p**2, dim=-1)
        
        q_new, p_new = q.clone(), p.clone()
        for _ in range(n_leapfrog_steps):
            q_new, p_new = leapfrog(q_new, p_new, step_size, log_prob_fn, compute_grad)
        
        new_log_prob = log_prob_fn(q_new)
        new_K = 0.5 * torch.sum(p_new**2, dim=-1)
        
        log_accept_ratio = new_log_prob - current_log_prob + current_K - new_K
        u = torch.log(torch.rand(n_chains, device=device))
        accept = u < log_accept_ratio
        
        q[accept] = q_new[accept]
        current_log_prob[accept] = new_log_prob[accept]
        accepts += accept.float()
        
        samples[i] = q.clone()

    accept_rate = (accepts / n_samples).mean().item()
    return samples, accept_rate

def compute_rhat(samples: torch.Tensor) -> float:
    """
    Computes Gelman-Rubin R-hat for the chains.
    samples: (n_samples, n_chains, dim)
    """
    n, m, d = samples.shape
    if m < 2:
        return 1.0
        
    chain_means = samples.mean(dim=0) # (m, d)
    overall_mean = chain_means.mean(dim=0) # (d,)
    
    B = n * torch.sum((chain_means - overall_mean)**2, dim=0) / (m - 1)
    
    chain_vars = torch.sum((samples - chain_means)**2, dim=0) / (n - 1) # (m, d)
    W = chain_vars.mean(dim=0) # (d,)
    
    var_plus = (n - 1) / n * W + B / n
    rhat = torch.sqrt(var_plus / (W + 1e-8))
    return rhat.max().item()

def compute_mcmc_ess(samples: torch.Tensor) -> float:
    """
    Computes Effective Sample Size (ESS) for MCMC chains using autocorrelation.
    samples: (n_samples, n_chains, dim)
    Returns the minimum ESS across dimensions.
    """
    n, m, d = samples.shape
    if n < 2:
        return 1.0
        
    # Compute mean across samples
    mean = samples.mean(dim=0) # (m, d)
    var = samples.var(dim=0) # (m, d)
    
    ess_dims = []
    for dim_idx in range(d):
        ess_chains = []
        for chain_idx in range(m):
            x = samples[:, chain_idx, dim_idx]
            # Autocovariance
            x_centered = (x - mean[chain_idx, dim_idx]).numpy()
            autocov = np.correlate(x_centered, x_centered, mode='full')
            # Extract positive lags
            autocov = autocov[n-1:] / n
            var_val = var[chain_idx, dim_idx].item()
            autocor = autocov / (var_val + 1e-8)
            
            # Sum correlations until first negative
            sum_rho = 0.0
            for k in range(1, n-1, 2):
                rho = autocor[k] + autocor[k+1]
                if rho < 0:
                    break
                sum_rho += rho
            
            tau = 1.0 + 2.0 * sum_rho
            ess_chains.append(n / tau)
        ess_dims.append(np.mean(ess_chains))
    
    return np.min(ess_dims)

def generate_reference_samples():
    from oracle_mimo import solve_lambda_star_mimo_radial, log_w_J_mimo
    from flow import RealNVPFlow
    
    configs = [
        (2, 8, 64),
        (4, 10, 128),
        (8, 12, 128)
    ]
    
    device = "cpu"
    sigma2 = 0.1
    A = 10.0
    P = A**2 / 9.0
    
    for n_t, n_couplings, hidden in configs:
        out_path = f"mcmc_reference_nt{n_t}.pt"
        if os.path.exists(out_path):
            print(f"Skipping n_t={n_t}, already exists")
            continue
            
        print(f"\nRunning MCMC for n_t={n_t}...")
        dim = 2 * n_t
        lambda_star = solve_lambda_star_mimo_radial(P, A, n_t, sigma2)
        
        def target_log_prob(theta):
            return log_w_J_mimo(theta, lambda_star, sigma2, n_t)
            
        # Initialize from flow to reduce warmup
        flow = RealNVPFlow(dim=dim, n_coupling_layers=n_couplings, hidden_dim=hidden)
        checkpoint_path = f"checkpoints/flow_mimo_nt{n_t}.pt"
        if os.path.exists(checkpoint_path):
            flow.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            flow.eval()
            with torch.no_grad():
                init_q = flow.sample(4) # 4 chains
        else:
            print("Checkpoint not found, initializing with standard normal")
            init_q = torch.randn(4, dim)
            
        n_samples = 5000
        n_warmup = 2000
        n_chains = 4
        
        samples, accept_rate = run_hmc(
            target_log_prob, dim, n_chains=n_chains, 
            n_samples=n_samples, n_warmup=n_warmup, 
            step_size=0.05, n_leapfrog_steps=10, 
            init_q=init_q, device=device
        )
        
        rhat = compute_rhat(samples)
        ess = compute_mcmc_ess(samples)
        
        print(f"n_t={n_t}: Accept rate = {accept_rate:.3f}, Max R-hat = {rhat:.3f}, Min ESS = {ess:.1f}")
        
        if rhat > 1.1:
            print(f"WARNING: R-hat {rhat:.3f} > 1.1 for n_t={n_t}. Consider increasing warmup.")
        
        # Trace plot
        plt.figure(figsize=(10, 4))
        for c in range(n_chains):
            plt.plot(samples[:, c, 0].numpy(), alpha=0.7, label=f"Chain {c}")
        plt.title(f"Trace plot of dim 0 (n_t={n_t})")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig(f"mcmc_trace_nt{n_t}.png")
        plt.close()
        
        # Flatten and save
        flat_samples = samples.reshape(-1, dim)
        torch.save(flat_samples, f"mcmc_reference_nt{n_t}.pt")
        print(f"Saved mcmc_reference_nt{n_t}.pt with {flat_samples.shape[0]} samples")

if __name__ == "__main__":
    generate_reference_samples()
