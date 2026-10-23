import torch
import numpy as np
import json
from config import FlowConfig, set_seed
from flow import RealNVPFlow
from diagnostics import compute_mmd_rbf, compute_energy_distance, sir_resample, compute_ess
from oracle_mimo import solve_lambda_star_mimo_radial, log_w_J_mimo
import os

def evaluate_nt(n_t: int, n_coupling_layers: int, hidden_dim: int, N: int = 5000):
    d = 2 * n_t
    sigma2 = 0.1
    A = 10.0
    P = A**2 / 9.0
    
    # Load MCMC reference samples
    mcmc_path = f"mcmc_reference_nt{n_t}.pt"
    if not os.path.exists(mcmc_path):
        print(f"MCMC reference not found for n_t={n_t} at {mcmc_path}")
        return None
    ref_samples_full = torch.load(mcmc_path, weights_only=True)
    
    flow = RealNVPFlow(dim=d, n_coupling_layers=n_coupling_layers, hidden_dim=hidden_dim)
    checkpoint_path = f"checkpoints/flow_mimo_nt{n_t}.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found for n_t={n_t} at {checkpoint_path}")
        return None
        
    flow.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    flow.eval()
    
    lambda_star = solve_lambda_star_mimo_radial(P, A, n_t, sigma2)
    
    def log_w_J_wrapper(theta):
        return log_w_J_mimo(theta, lambda_star, sigma2, n_t)
        
    mmds = []
    energy_dists = []
    ess_ns = []
    
    seeds = [42, 43, 44, 45, 46]
    for seed in seeds:
        set_seed(seed)
        
        with torch.no_grad():
            samples = flow.sample(N)
            log_q = flow.log_prob(samples)
            log_w = log_w_J_wrapper(samples)
            
            ess, ess_n = compute_ess(samples, log_w, log_q)
            ess_ns.append(ess_n)
            
            # Subsample MCMC to match flow size if larger
            if ref_samples_full.shape[0] > N:
                idx = torch.randperm(ref_samples_full.shape[0])[:N]
                ref_samples = ref_samples_full[idx]
            else:
                ref_samples = ref_samples_full
                
            mmd = compute_mmd_rbf(samples, ref_samples, bandwidth='median')
            energy_dist = compute_energy_distance(samples, ref_samples)
            
            mmds.append(mmd)
            energy_dists.append(energy_dist)
            
    return {
        "n_t": n_t,
        "d": d,
        "ESS_N_mean": np.mean(ess_ns),
        "ESS_N_std": np.std(ess_ns),
        "MMD_mean": np.mean(mmds),
        "MMD_std": np.std(mmds),
        "Energy_Distance_mean": np.mean(energy_dists),
        "Energy_Distance_std": np.std(energy_dists)
    }

def main():
    results = []
    
    configs = [
        (2, 8, 64),
        (4, 10, 128),
        (8, 12, 128)
    ]
    
    for nt, n_couplings, hidden in configs:
        print(f"Evaluating n_t={nt}...")
        res = evaluate_nt(nt, n_couplings, hidden)
        if res:
            results.append(res)
            print(f"  ESS/N: {res['ESS_N_mean']:.6f} ± {res['ESS_N_std']:.6f}")
            print(f"  MMD: {res['MMD_mean']:.6f} ± {res['MMD_std']:.6f}")
            print(f"  Energy Dist: {res['Energy_Distance_mean']:.6f} ± {res['Energy_Distance_std']:.6f}")
            
    with open("evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Saved evaluation_metrics.json")

if __name__ == "__main__":
    main()
