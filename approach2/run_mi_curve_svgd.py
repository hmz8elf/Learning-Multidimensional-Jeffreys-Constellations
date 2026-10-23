import os
import json
import time
import gc
import torch
torch.set_num_threads(2)

import numpy as np
from itertools import product
from mi_curve import compute_mi_ba, M_VALUES, SIGMA2, N_Y, N_SEEDS, BA_MAX_ITER, BA_TOL, RESULTS_FILE, check_memory_before, save_results

from svgd import svgd_sample
from oracle_mimo import solve_lambda_star_mimo_radial, log_w_J_mimo
from config import set_seed

# Les params globaux
A = 10.0
P = A**2 / 9.0

def run_svgd_mi():
    with open(RESULTS_FILE, "r") as f:
        all_results = json.load(f)
        
    for nt in [2, 4, 8]:
        d = 2 * nt
        key = f"nt{nt}"
        
        if key not in all_results:
            continue
            
        print(f"\n============================================================")
        print(f"SVGD MI Curve - nt={nt} (d={d})")
        print(f"============================================================")
        
        if "svgd" not in all_results[key]:
            all_results[key]["svgd"] = {}
            
        lambda_star = solve_lambda_star_mimo_radial(P, A, nt, SIGMA2)
        def log_w_J_wrapper(theta):
            return log_w_J_mimo(theta, lambda_star, SIGMA2, nt)
            
        for M in M_VALUES:
            print(f"\n  M={M}")
            mi_seeds = []
            
            already_done = all_results[key]["svgd"].get(str(M), {}).get("seeds", [])
            
            for seed in range(N_SEEDS):
                if seed < len(already_done):
                    mi = already_done[seed]
                    print(f"    SVGD  seed={seed}: MI={mi:.4f} bits [cached]")
                    mi_seeds.append(mi)
                    continue

                set_seed(seed + 200 * M + 20000 * nt)
                rng = np.random.default_rng(seed + 200 * M + 20000 * nt)
                
                # SVGD takes some time. n_steps=500
                t_s = time.time()
                # Run SVGD for exactly M particles! (For MI curve, the constellation size is M)
                # Wait, does SVGD work with just M particles (e.g. M=4)?
                # With M=4, median distance heuristic might be weird, but let's see.
                X_svgd = svgd_sample(log_w_J_wrapper, n_particles=M, n_steps=500, dim=d, lr=0.5)
                X_svgd = X_svgd.cpu().numpy()
                
                t_ba = time.time()
                mi = compute_mi_ba(X_svgd, SIGMA2, N_Y, rng)
                mi_seeds.append(mi)
                print(f"    SVGD  seed={seed}: MI={mi:.4f} bits (SVGD {t_ba-t_s:.1f}s, BA {time.time()-t_ba:.1f}s)")
                
                all_results[key]["svgd"][str(M)] = {
                    "mean": float(np.mean(mi_seeds)),
                    "std": float(np.std(mi_seeds)),
                    "seeds": mi_seeds[:]
                }
                save_results(all_results)
                
            mean_mi = float(np.mean(mi_seeds))
            std_mi = float(np.std(mi_seeds))
            print(f"    SVGD → {mean_mi:.4f} ± {std_mi:.4f} bits")
            
        # Liberate
        gc.collect()

if __name__ == "__main__":
    run_svgd_mi()
