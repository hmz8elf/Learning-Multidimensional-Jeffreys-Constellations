import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from config import set_seed
from flow import RealNVPFlow
from diagnostics import compute_mmd_rbf, compute_energy_distance

def run_tests():
    configs = [
        (2, 8, 64),
        (4, 10, 128),
        (8, 12, 128)
    ]
    
    device = "cpu"
    N = 5000
    
    results = {}
    
    for n_t, n_couplings, hidden in configs:
        print(f"\n--- Processing n_t = {n_t} ---")
        dim = 2 * n_t
        
        # Load MCMC samples (Paquet B)
        mcmc_path = f"mcmc_reference_nt{n_t}.pt"
        if not os.path.exists(mcmc_path):
            print(f"MCMC reference not found for n_t={n_t}")
            continue
        ref_samples_full = torch.load(mcmc_path, map_location=device, weights_only=True)
        if ref_samples_full.shape[0] > N:
            idx = torch.randperm(ref_samples_full.shape[0])[:N]
            paquet_B = ref_samples_full[idx]
        else:
            paquet_B = ref_samples_full
            
        # Load Flow and sample (Paquet A)
        flow = RealNVPFlow(dim=dim, n_coupling_layers=n_couplings, hidden_dim=hidden)
        checkpoint_path = f"checkpoints/flow_mimo_nt{n_t}.pt"
        flow.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        flow.eval()
        with torch.no_grad():
            paquet_A = flow.sample(N)
            
        # 1. BASELINE DE BRUIT INTERNE (MCMC vs MCMC)
        print("Computing Internal Noise Baseline...")
        internal_mmds = []
        internal_eds = []
        # We split the full MCMC samples (which has 20000 samples) into two halves of size N
        # Wait, the prompt says: "Prends uniquement le paquet B (MCMC, la référence), sans le flow. Coupe-le en deux moitiés aléatoires de taille égale."
        # If we take paquet B of size N, we split it into N/2 and N/2. Let's do that.
        N_half = paquet_B.shape[0] // 2
        
        # We need bandwidth for MMD. We can use median heuristic on A and B.
        Z_for_bw = torch.cat([paquet_A[:1000], paquet_B[:1000]], dim=0)
        dists_for_bw = torch.pdist(Z_for_bw)
        bandwidth = torch.median(dists_for_bw).item()
        if bandwidth < 1e-5: bandwidth = 1.0
        print(f"Bandwidth: {bandwidth}")
        
        for _ in range(10):
            idx_b = torch.randperm(paquet_B.shape[0])
            half1 = paquet_B[idx_b[:N_half]]
            half2 = paquet_B[idx_b[N_half:2*N_half]]
            
            # Use compute_mmd_rbf but force the bandwidth so it's consistent
            # Actually our compute_mmd_rbf uses bandwidth='median' by default. We can pass the float.
            internal_mmds.append(compute_mmd_rbf(half1, half2, bandwidth=bandwidth))
            internal_eds.append(compute_energy_distance(half1, half2))
            
        baseline_mmd_mean, baseline_mmd_std = np.mean(internal_mmds), np.std(internal_mmds)
        baseline_ed_mean, baseline_ed_std = np.mean(internal_eds), np.std(internal_eds)
        print(f"Internal Baseline MMD: {baseline_mmd_mean:.6f} +/- {baseline_mmd_std:.6f}")
        print(f"Internal Baseline ED:  {baseline_ed_mean:.6f} +/- {baseline_ed_std:.6f}")
        
        # 2. TEST DE PERMUTATION
        print("Computing Permutation Test...")
        # Vraie valeur observée
        obs_mmd = compute_mmd_rbf(paquet_A, paquet_B, bandwidth=bandwidth)
        obs_ed = compute_energy_distance(paquet_A, paquet_B)
        print(f"Observed MMD: {obs_mmd:.6f}, Observed ED: {obs_ed:.6f}")
        
        Z = torch.cat([paquet_A, paquet_B], dim=0)
        N_Z = Z.shape[0]
        N_A = paquet_A.shape[0]
        
        # Precompute distance matrix and kernel matrix to speed up 1000 permutations
        # cdist is memory intensive for 10000x10000 (400MB). It fits in RAM easily.
        D = torch.cdist(Z, Z)
        K = torch.exp(- (D**2) / (2 * bandwidth**2))
        
        perm_mmds = []
        perm_eds = []
        
        for _ in range(1000):
            idx = torch.randperm(N_Z)
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
        
        # p-value
        p_val_mmd = np.mean(perm_mmds >= obs_mmd)
        p_val_ed = np.mean(perm_eds >= obs_ed)
        print(f"P-value MMD: {p_val_mmd:.4f}, P-value ED: {p_val_ed:.4f}")
        
        results[n_t] = {
            "obs_mmd": obs_mmd,
            "obs_ed": obs_ed,
            "baseline_mmd_mean": baseline_mmd_mean,
            "baseline_mmd_std": baseline_mmd_std,
            "baseline_ed_mean": baseline_ed_mean,
            "baseline_ed_std": baseline_ed_std,
            "p_val_mmd": p_val_mmd,
            "p_val_ed": p_val_ed
        }
        
        # 3. Figures
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(perm_mmds, bins=30, color='blue', alpha=0.7)
        plt.axvline(obs_mmd, color='red', linestyle='dashed', linewidth=2, label=f'Observed\n(p={p_val_mmd:.3f})')
        plt.title(f"MMD Permutation Test (n_t={n_t})")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(perm_eds, bins=30, color='green', alpha=0.7)
        plt.axvline(obs_ed, color='red', linestyle='dashed', linewidth=2, label=f'Observed\n(p={p_val_ed:.3f})')
        plt.title(f"ED Permutation Test (n_t={n_t})")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"permutation_test_nt{n_t}.png")
        plt.close()
        
    with open("permutation_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Done! Results saved in permutation_results.json")

if __name__ == "__main__":
    set_seed(42)
    run_tests()
