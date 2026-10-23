import torch
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from flow import RealNVPFlow

def measure_time(flow, d, batch_size=1000, n_warmup=10, n_repeats=20):
    device = next(flow.parameters()).device
    z = torch.randn(batch_size, d, device=device)
    for _ in range(n_warmup):
        _ = flow.forward_full(z)
        
    times = []
    for _ in range(n_repeats):
        z = torch.randn(batch_size, d, device=device)
        start = time.perf_counter()
        _ = flow.forward_full(z)
        end = time.perf_counter()
        times.append(end - start)
        
    return np.mean(times), np.std(times)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    d_fixed = 8
    
    configs = [
        {"L": 8, "h": 64},   # Matches nt=2 architecture
        {"L": 10, "h": 128}, # Matches nt=4 architecture
        {"L": 12, "h": 128}  # Matches nt=8 architecture
    ]
    
    results = {}
    
    L_vals = []
    times = []
    stds = []
    
    for cfg in configs:
        L = cfg["L"]
        h = cfg["h"]
        flow = RealNVPFlow(dim=d_fixed, n_coupling_layers=L, hidden_dim=h).to(device)
        flow.eval()
        with torch.no_grad():
            t_mean, t_std = measure_time(flow, d_fixed)
            
        key = f"synth_L{L}_h{h}_d{d_fixed}"
        results[key] = {"mean": t_mean, "std": t_std, "L": L, "h": h, "d": d_fixed}
        L_vals.append(L)
        times.append(t_mean)
        stds.append(t_std)
        print(f"Fixed d={d_fixed}, L={L}, h={h}: {t_mean*1000:.3f} ± {t_std*1000:.3f} ms")
        
    # Read existing JSON to update it
    try:
        with open("benchmark_logdet_complexity.json", "r") as f:
            existing_results = json.load(f)
    except FileNotFoundError:
        existing_results = {}
        
    existing_results.update(results)
    
    with open("benchmark_logdet_complexity.json", "w") as f:
        json.dump(existing_results, f, indent=4)
        
    # Plotting: time vs L
    plt.figure(figsize=(8, 5))
    plt.errorbar(L_vals, [t*1000 for t in times], yerr=[s*1000 for s in stds], fmt='o-', color='green', label=f'Synthetic Architectures (Fixed d={d_fixed})')
    
    # Also grab the trained models times from existing json for comparison
    trained_L = [8, 10, 12]
    trained_times = [
        existing_results.get("trained_d4", {}).get("mean", 0) * 1000,
        existing_results.get("trained_d8", {}).get("mean", 0) * 1000,
        existing_results.get("trained_d16", {}).get("mean", 0) * 1000,
    ]
    trained_stds = [
        existing_results.get("trained_d4", {}).get("std", 0) * 1000,
        existing_results.get("trained_d8", {}).get("std", 0) * 1000,
        existing_results.get("trained_d16", {}).get("std", 0) * 1000,
    ]
    
    if all(trained_times):
        plt.errorbar(trained_L, trained_times, yerr=trained_stds, fmt='s--', color='red', label='Trained Models (Varying d)')
        
    plt.xlabel('Number of Coupling Layers (L)')
    plt.ylabel('Time (ms) for Forward log-det (Batch=1000)')
    plt.title(f'Log-Det Computation Time vs Number of Layers (Fixed d={d_fixed})')
    plt.grid(True, which="both", ls="--")
    plt.xticks(L_vals)
    plt.legend()
    
    plt.savefig('benchmark_L_isolation.png', dpi=300, bbox_inches='tight')
    print("Saved benchmark_L_isolation.png")

if __name__ == "__main__":
    main()
