import torch
torch.set_num_threads(2)

import numpy as np
import json
import os
from config import set_seed
from svgd import svgd_sample
from diagnostics import compute_mmd_rbf, compute_energy_distance
from oracle_mimo import solve_lambda_star_mimo_radial, log_w_J_mimo
import time

def evaluate_nt_svgd(n_t: int, n_samples: int = 1000, n_steps: int = 500):
    d = 2 * n_t
    sigma2 = 0.1
    A = 10.0
    P = A**2 / 9.0
    
    mcmc_path = f"mcmc_reference_nt{n_t}.pt"
    if not os.path.exists(mcmc_path):
        print(f"MCMC reference not found for n_t={n_t} at {mcmc_path}")
        return None
    ref_samples_full = torch.load(mcmc_path, weights_only=True)
    
    lambda_star = solve_lambda_star_mimo_radial(P, A, n_t, sigma2)
    
    def log_w_J_wrapper(theta):
        return log_w_J_mimo(theta, lambda_star, sigma2, n_t)
        
    mmds = []
    energy_dists = []
    
    # We use fewer seeds for SVGD to save time, e.g., 3 seeds.
    seeds = [42, 43, 44]
    
    for seed in seeds:
        set_seed(seed)
        
        t0 = time.time()
        print(f"    Running SVGD for nt={n_t}, seed={seed}...")
        # Use a slightly lower learning rate or 0.5
        samples = svgd_sample(log_w_J_wrapper, n_samples, n_steps, dim=d, lr=0.5)
        print(f"    Done in {time.time()-t0:.1f}s")
        
        if ref_samples_full.shape[0] > n_samples:
            idx = torch.randperm(ref_samples_full.shape[0])[:n_samples]
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
        "ESS_N_mean": "N/A",
        "ESS_N_std": "N/A",
        "MMD_mean": float(np.mean(mmds)),
        "MMD_std": float(np.std(mmds)),
        "Energy_Distance_mean": float(np.mean(energy_dists)),
        "Energy_Distance_std": float(np.std(energy_dists))
    }

def main():
    results = []
    
    # nt = 2, 4, 8
    configs = [2, 4, 8]
    
    for nt in configs:
        print(f"Evaluating SVGD n_t={nt}...")
        # Liberate memory explicitly
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        res = evaluate_nt_svgd(nt)
        if res:
            results.append(res)
            print(f"  MMD: {res['MMD_mean']:.6f} ± {res['MMD_std']:.6f}")
            print(f"  Energy Dist: {res['Energy_Distance_mean']:.6f} ± {res['Energy_Distance_std']:.6f}")
            
    with open("svgd_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Saved svgd_metrics.json")

if __name__ == "__main__":
    main()
