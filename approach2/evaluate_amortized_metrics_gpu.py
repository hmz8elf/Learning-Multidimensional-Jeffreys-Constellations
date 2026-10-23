import torch
import numpy as np
import json
import os
import gc
from config import set_seed
from flow import RealNVPFlow
from diagnostics import compute_mmd_rbf, compute_energy_distance
from oracle_mimo import solve_lambda_star_mimo_radial, log_w_J_mimo
from amortized_mcmc import load_flow, amortized_mh_sample

def evaluate_nt(n_t: int, device: torch.device, N: int = 5000):
    d = 2 * n_t
    sigma2 = 0.1
    A = 10.0
    P = A**2 / 9.0
    
    mcmc_path = f"mcmc_reference_nt{n_t}.pt"
    if not os.path.exists(mcmc_path):
        print(f"MCMC reference not found for n_t={n_t} at {mcmc_path}")
        return None
    ref_samples_full = torch.load(mcmc_path, weights_only=True, map_location=device)
    
    flow = load_flow(n_t, device)
    lambda_star = solve_lambda_star_mimo_radial(P, A, n_t, sigma2)
    
    def log_w_J_wrapper(theta):
        return log_w_J_mimo(theta, lambda_star, sigma2, n_t)
        
    mmds = []
    energy_dists = []
    accept_rates = []
    rhats = []
    
    seeds = [42, 43, 44, 45, 46]
    for seed in seeds:
        set_seed(seed)
        
        samples, ar, rhat = amortized_mh_sample(
            log_w_J_fn=log_w_J_wrapper,
            flow=flow,
            n_samples=N,
            n_chains=4,
            n_warmup=500,
            device=device,
        )
        
        accept_rates.append(ar)
        rhats.append(rhat)
        
        if ref_samples_full.shape[0] > N:
            idx = torch.randperm(ref_samples_full.shape[0], device=device)[:N]
            ref_samples = ref_samples_full[idx]
        else:
            ref_samples = ref_samples_full
            
        mmd = compute_mmd_rbf(samples, ref_samples, bandwidth='median')
        energy_dist = compute_energy_distance(samples, ref_samples)
        
        mmds.append(mmd)
        energy_dists.append(energy_dist)
        
    # Free memory
    del flow
    del ref_samples_full
    torch.cuda.empty_cache()
    gc.collect()
        
    return {
        "n_t": n_t,
        "d": d,
        "Accept_Rate_mean": np.mean(accept_rates),
        "Accept_Rate_std": np.std(accept_rates),
        "Rhat_max_mean": np.mean(rhats),
        "MMD_mean": np.mean(mmds),
        "MMD_std": np.std(mmds),
        "Energy_Distance_mean": np.mean(energy_dists),
        "Energy_Distance_std": np.std(energy_dists)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = []
    
    for nt in [2, 4, 8]:
        print(f"Evaluating n_t={nt} on {device}...")
        res = evaluate_nt(nt, device)
        if res:
            results.append(res)
            print(f"  Accept Rate: {res['Accept_Rate_mean']:.4f} ± {res['Accept_Rate_std']:.4f}")
            print(f"  R-hat Max: {res['Rhat_max_mean']:.4f}")
            print(f"  MMD: {res['MMD_mean']:.6f} ± {res['MMD_std']:.6f}")
            print(f"  Energy Dist: {res['Energy_Distance_mean']:.6f} ± {res['Energy_Distance_std']:.6f}")
            
    with open("amortized_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Saved amortized_metrics.json")

if __name__ == "__main__":
    main()
