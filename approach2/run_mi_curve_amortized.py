import os
import json
import time
import gc
import torch
torch.set_num_threads(2)

import numpy as np
from itertools import product
from mi_curve import compute_mi_ba, M_VALUES, SIGMA2, N_Y, N_SEEDS, BA_MAX_ITER, BA_TOL, RESULTS_FILE, check_memory_before, save_results

from amortized_mcmc import load_flow, amortized_mh_sample
from oracle_mimo import solve_lambda_star_mimo_radial, log_w_J_mimo
from config import set_seed

A = 10.0
P = A**2 / 9.0

def run_amortized_mi():
    with open(RESULTS_FILE, "r") as f:
        all_results = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    for nt in [2, 4, 8]:
        d = 2 * nt
        key = f"nt{nt}"
        
        if key not in all_results:
            continue
            
        print(f"\n============================================================")
        print(f"Amortized MCMC MI Curve - nt={nt} (d={d})")
        print(f"============================================================")
        
        if "amortized" not in all_results[key]:
            all_results[key]["amortized"] = {}
            
        lambda_star = solve_lambda_star_mimo_radial(P, A, nt, SIGMA2)
        def log_w_J_wrapper(theta):
            return log_w_J_mimo(theta, lambda_star, SIGMA2, nt)
            
        flow = load_flow(nt, device)
            
        for M in M_VALUES:
            print(f"\n  M={M}")
            mi_seeds = []
            
            already_done = all_results[key]["amortized"].get(str(M), {}).get("seeds", [])
            
            for seed in range(N_SEEDS):
                if seed < len(already_done):
                    mi = already_done[seed]
                    print(f"    Amortized  seed={seed}: MI={mi:.4f} bits [cached]")
                    mi_seeds.append(mi)
                    continue

                set_seed(seed + 400 * M + 40000 * nt)
                rng = np.random.default_rng(seed + 400 * M + 40000 * nt)
                
                t_s = time.time()
                # Run amortized MCMC for M particles
                X_amortized, _, _ = amortized_mh_sample(
                    log_w_J_fn=log_w_J_wrapper, 
                    flow=flow,
                    n_samples=M, 
                    n_chains=4,
                    n_warmup=500,
                    device=device
                )
                X_amortized = X_amortized.cpu().numpy()
                
                t_ba = time.time()
                mi = compute_mi_ba(X_amortized, SIGMA2, N_Y, rng)
                mi_seeds.append(mi)
                print(f"    Amortized  seed={seed}: MI={mi:.4f} bits (Amortized {t_ba-t_s:.1f}s, BA {time.time()-t_ba:.1f}s)")
                
                all_results[key]["amortized"][str(M)] = {
                    "mean": float(np.mean(mi_seeds)),
                    "std": float(np.std(mi_seeds)),
                    "seeds": mi_seeds[:]
                }
                save_results(all_results)
                
            mean_mi = float(np.mean(mi_seeds))
            std_mi = float(np.std(mi_seeds))
            print(f"    Amortized → {mean_mi:.4f} ± {std_mi:.4f} bits")
            
        del flow
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_amortized_mi()
