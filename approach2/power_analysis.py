import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import psutil
from config import set_seed
from flow import RealNVPFlow
from diagnostics import compute_mmd_rbf, compute_energy_distance

def run_power_analysis():
    # Hardware limits to avoid overutilization
    torch.set_num_threads(4) # Limit CPU threads
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Limit GPU memory usage to 30% to not freeze the desktop
        try:
            torch.cuda.set_per_process_memory_fraction(0.3, 0)
        except Exception as e:
            print("Could not set memory fraction:", e)
            
    print(f"Running on device: {device}")
    
    configs = [
        (2, 8, 64),
        (4, 10, 128),
        (8, 12, 128)
    ]
    
    N = 5000
    perturbations = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    
    results = {}
    
    for n_t, n_couplings, hidden in configs:
        print(f"\n--- Power Analysis n_t = {n_t} ---")
        dim = 2 * n_t
        
        # Load MCMC samples (Paquet B)
        mcmc_path = f"mcmc_reference_nt{n_t}.pt"
        if not os.path.exists(mcmc_path):
            print(f"MCMC reference not found for n_t={n_t}")
            continue
        ref_samples_full = torch.load(mcmc_path, map_location=device, weights_only=True)
        if ref_samples_full.shape[0] > N:
            idx = torch.randperm(ref_samples_full.shape[0], device=device)[:N]
            paquet_B = ref_samples_full[idx]
        else:
            paquet_B = ref_samples_full
            
        # Load Flow and sample (Paquet A)
        flow = RealNVPFlow(dim=dim, n_coupling_layers=n_couplings, hidden_dim=hidden).to(device)
        checkpoint_path = f"checkpoints/flow_mimo_nt{n_t}.pt"
        flow.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        flow.eval()
        with torch.no_grad():
            paquet_A_orig = flow.sample(N)
            
        # Standard deviation of flow samples per dimension
        std_A = paquet_A_orig.std(dim=0)
        
        results[n_t] = {}
        
        p_vals_mmd = []
        p_vals_ed = []
        
        for pert in perturbations:
            print(f"  Testing perturbation: {pert*100:.0f}%")
            
            # Apply perturbation
            paquet_A = paquet_A_orig + pert * std_A
            
            # We need bandwidth for MMD
            Z_for_bw = torch.cat([paquet_A[:1000], paquet_B[:1000]], dim=0)
            dists_for_bw = torch.pdist(Z_for_bw)
            bandwidth = torch.median(dists_for_bw).item()
            if bandwidth < 1e-5: bandwidth = 1.0
            
            # Note: compute_mmd_rbf and compute_energy_distance expect CPU tensors in some previous files maybe? 
            # Let's ensure they run on device.
            # Z is N_A + N_B
            Z = torch.cat([paquet_A, paquet_B], dim=0)
            N_Z = Z.shape[0]
            N_A = paquet_A.shape[0]
            
            # Compute distance matrix directly on GPU (400MB)
            D = torch.cdist(Z, Z)
            K = torch.exp(- (D**2) / (2 * bandwidth**2))
            
            # Observed values
            k_xx_obs = K[:N_A, :N_A].mean().item()
            k_yy_obs = K[N_A:, N_A:].mean().item()
            k_xy_obs = K[:N_A, N_A:].mean().item()
            obs_mmd = k_xx_obs + k_yy_obs - 2 * k_xy_obs
            
            d_xx_obs = D[:N_A, :N_A].mean().item()
            d_yy_obs = D[N_A:, N_A:].mean().item()
            d_xy_obs = D[:N_A, N_A:].mean().item()
            obs_ed = 2 * d_xy_obs - d_xx_obs - d_yy_obs
            
            perm_mmds = []
            perm_eds = []
            
            for _ in range(1000):
                idx = torch.randperm(N_Z, device=device)
                idx_A = idx[:N_A]
                idx_B = idx[N_A:]
                
                # MMD
                k_xx = K[idx_A][:, idx_A].mean().item()
                k_yy = K[idx_B][:, idx_B].mean().item()
                k_xy = K[idx_A][:, idx_B].mean().item()
                perm_mmd = k_xx + k_yy - 2 * k_xy
                perm_mmds.append(perm_mmd)
                
                # ED
                d_xx = D[idx_A][:, idx_A].mean().item()
                d_yy = D[idx_B][:, idx_B].mean().item()
                d_xy = D[idx_A][:, idx_B].mean().item()
                perm_ed = 2 * d_xy - d_xx - d_yy
                perm_eds.append(perm_ed)
                
            perm_mmds = np.array(perm_mmds)
            perm_eds = np.array(perm_eds)
            
            p_val_mmd = np.mean(perm_mmds >= obs_mmd)
            p_val_ed = np.mean(perm_eds >= obs_ed)
            
            p_vals_mmd.append(p_val_mmd)
            p_vals_ed.append(p_val_ed)
            
            results[n_t][str(pert)] = {
                "p_val_mmd": p_val_mmd,
                "p_val_ed": p_val_ed
            }
            
            # Clear CUDA cache periodically to avoid fragmentation
            if device == "cuda":
                torch.cuda.empty_cache()
            
        # Plotting
        # Move back to CPU for plotting just in case
        plt.figure(figsize=(8, 5))
        plt.plot([p * 100 for p in perturbations], p_vals_mmd, marker='o', label='MMD p-value')
        plt.plot([p * 100 for p in perturbations], p_vals_ed, marker='s', label='Energy Dist p-value')
        plt.axhline(y=0.05, color='r', linestyle='--', label='Significance threshold (0.05)')
        plt.xlabel("Perturbation Amplitude (% of std)")
        plt.ylabel("p-value")
        plt.title(f"Statistical Power Analysis (n_t={n_t})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"power_analysis_nt{n_t}.png")
        plt.close()

    with open("power_analysis_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Done! Results saved in power_analysis_results.json")

if __name__ == "__main__":
    set_seed(42)
    run_power_analysis()
