import torch
import numpy as np
import json
import os
from flow import RealNVPFlow

def run_anomaly_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    n_t = 4
    d = 8
    L = 10
    h = 128
    N = 5000
    
    # Load MCMC samples
    mcmc_path = f"mcmc_reference_nt{n_t}.pt"
    ref_samples_full = torch.load(mcmc_path, map_location=device, weights_only=True)
    
    # Load Flow
    flow = RealNVPFlow(dim=d, n_coupling_layers=L, hidden_dim=h).to(device)
    flow.load_state_dict(torch.load(f"checkpoints/flow_mimo_nt{n_t}.pt", map_location=device, weights_only=True))
    flow.eval()
    
    # We will do 10 iterations. 
    # To be extremely rigorous and cover both interpretations of "bruit":
    # We will apply a RANDOM SHIFT (random direction) and RANDOM NOISE (per particle)
    # But wait, let's just do random noise as requested: "génération du bruit de perturbation"
    
    results = {"1%": [], "2%": []}
    
    for pert in [0.01, 0.02]:
        print(f"\n--- Testing perturbation: {pert*100:.0f}% ---")
        p_vals_mmd = []
        p_vals_ed = []
        
        for seed in range(10):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Resample A and B for strict independence of the 10 trials
            if ref_samples_full.shape[0] > N:
                idx = torch.randperm(ref_samples_full.shape[0], device=device)[:N]
                paquet_B = ref_samples_full[idx]
            else:
                paquet_B = ref_samples_full
                
            with torch.no_grad():
                paquet_A_orig = flow.sample(N)
                
            std_A = paquet_A_orig.std(dim=0)
            
            # The prompt says: "10 seeds différentes pour la génération du bruit de perturbation."
            # The original script did: paquet_A_orig + pert * std_A (deterministic shift vector of ones * pert * std)
            # Let's apply random Gaussian noise of scale pert * std_A
            noise = torch.randn_like(paquet_A_orig) * pert * std_A
            paquet_A = paquet_A_orig + noise
            
            # Compute MMD and ED p-values
            Z = torch.cat([paquet_A, paquet_B], dim=0)
            N_A = N
            N_Z = 2 * N
            
            # distance matrix
            D = torch.cdist(Z, Z)
            
            # bandwidth
            dists_for_bw = D[:1000, 1000:2000].flatten() # approximate median
            bandwidth = torch.median(dists_for_bw).item()
            if bandwidth < 1e-5: bandwidth = 1.0
            
            K = torch.exp(- (D**2) / (2 * bandwidth**2))
            
            obs_mmd = K[:N_A, :N_A].mean().item() + K[N_A:, N_A:].mean().item() - 2 * K[:N_A, N_A:].mean().item()
            obs_ed = 2 * D[:N_A, N_A:].mean().item() - D[:N_A, :N_A].mean().item() - D[N_A:, N_A:].mean().item()
            
            perm_mmds = []
            perm_eds = []
            
            for _ in range(1000):
                idx = torch.randperm(N_Z, device=device)
                idx_A = idx[:N_A]
                idx_B = idx[N_A:]
                
                k_xx = K[idx_A][:, idx_A].mean().item()
                k_yy = K[idx_B][:, idx_B].mean().item()
                k_xy = K[idx_A][:, idx_B].mean().item()
                perm_mmds.append(k_xx + k_yy - 2 * k_xy)
                
                d_xx = D[idx_A][:, idx_A].mean().item()
                d_yy = D[idx_B][:, idx_B].mean().item()
                d_xy = D[idx_A][:, idx_B].mean().item()
                perm_eds.append(2 * d_xy - d_xx - d_yy)
                
            p_val_mmd = np.mean(np.array(perm_mmds) >= obs_mmd)
            p_val_ed = np.mean(np.array(perm_eds) >= obs_ed)
            
            p_vals_mmd.append(p_val_mmd)
            p_vals_ed.append(p_val_ed)
            
            print(f"  Seed {seed}: MMD p-value = {p_val_mmd:.3f}, ED p-value = {p_val_ed:.3f}")
            
            if device == "cuda":
                torch.cuda.empty_cache()
                
        results[f"{pert*100:.0f}%"] = {"mmd": p_vals_mmd, "ed": p_vals_ed}
        
    with open("nt4_anomaly_check.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_anomaly_check()
