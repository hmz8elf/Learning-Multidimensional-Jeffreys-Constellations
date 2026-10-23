import torch
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from flow import RealNVPFlow
import os

def measure_time(flow, d, batch_size=1000, n_warmup=10, n_repeats=20):
    device = next(flow.parameters()).device
    # Generate batch
    z = torch.randn(batch_size, d, device=device)
    
    # Warmup
    for _ in range(n_warmup):
        _ = flow.forward_full(z)
        
    times = []
    for _ in range(n_repeats):
        z = torch.randn(batch_size, d, device=device)
        start = time.perf_counter()
        # forward_full returns (theta, log_det)
        _ = flow.forward_full(z)
        end = time.perf_counter()
        times.append(end - start)
        
    return np.mean(times), np.std(times)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    results = {}
    
    # Trained models
    trained_configs = [
        {"nt": 2, "d": 4, "L": 8, "h": 64, "ckpt": "checkpoints/flow_mimo_nt2.pt"},
        {"nt": 4, "d": 8, "L": 10, "h": 128, "ckpt": "checkpoints/flow_mimo_nt4.pt"},
        {"nt": 8, "d": 16, "L": 12, "h": 128, "ckpt": "checkpoints/flow_mimo_nt8.pt"}
    ]
    
    dims_trained = []
    times_trained = []
    stds_trained = []
    
    for cfg in trained_configs:
        d = cfg["d"]
        L = cfg["L"]
        h = cfg["h"]
        flow = RealNVPFlow(dim=d, n_coupling_layers=L, hidden_dim=h).to(device)
        
        if os.path.exists(cfg["ckpt"]):
            flow.load_state_dict(torch.load(cfg["ckpt"], map_location=device))
            print(f"Loaded {cfg['ckpt']}")
        else:
            print(f"Warning: {cfg['ckpt']} not found! Using random initialization.")
            
        flow.eval()
        with torch.no_grad():
            t_mean, t_std = measure_time(flow, d)
            
        results[f"trained_d{d}"] = {"mean": t_mean, "std": t_std, "type": "trained", "L": L, "h": h}
        dims_trained.append(d)
        times_trained.append(t_mean)
        stds_trained.append(t_std)
        print(f"Trained d={d}: {t_mean*1000:.3f} ± {t_std*1000:.3f} ms")

    # Synthetic models with fixed architecture (L=8, h=64)
    synthetic_dims = [2, 4, 6, 8, 12, 16, 20, 32, 64]
    L_fixed = 8
    h_fixed = 64
    
    dims_synth = []
    times_synth = []
    stds_synth = []
    
    for d in synthetic_dims:
        flow = RealNVPFlow(dim=d, n_coupling_layers=L_fixed, hidden_dim=h_fixed).to(device)
        flow.eval()
        with torch.no_grad():
            t_mean, t_std = measure_time(flow, d)
            
        results[f"synth_d{d}"] = {"mean": t_mean, "std": t_std, "type": "synthetic", "L": L_fixed, "h": h_fixed}
        dims_synth.append(d)
        times_synth.append(t_mean)
        stds_synth.append(t_std)
        print(f"Synthetic d={d}: {t_mean*1000:.3f} ± {t_std*1000:.3f} ms")
        
    # Fit power law for synthetic models
    log_d_synth = np.log(dims_synth)
    log_t_synth = np.log(times_synth)
    slope_synth, intercept_synth = np.polyfit(log_d_synth, log_t_synth, 1)
    
    # Fit power law for trained models
    log_d_trained = np.log(dims_trained)
    log_t_trained = np.log(times_trained)
    slope_trained, intercept_trained = np.polyfit(log_d_trained, log_t_trained, 1)
    
    results["fit_synth"] = {"b": slope_synth, "log_a": intercept_synth}
    results["fit_trained"] = {"b": slope_trained, "log_a": intercept_trained}
    
    with open("benchmark_logdet_complexity.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot synthetic
    plt.errorbar(dims_synth, times_synth, yerr=stds_synth, fmt='o-', color='blue', label=f'Synthetic (Fixed Arch: L={L_fixed}, h={h_fixed})')
    
    # Plot trained
    plt.errorbar(dims_trained, times_trained, yerr=stds_trained, fmt='s--', color='red', label='Trained Models (Varying Arch)')
    
    # Plot fit lines
    d_range = np.linspace(min(synthetic_dims), max(synthetic_dims), 100)
    plt.plot(d_range, np.exp(intercept_synth) * d_range**slope_synth, 'b:', alpha=0.5, label=f'Fit Synth: b={slope_synth:.2f}')
    
    d_range_trained = np.linspace(min(dims_trained), max(dims_trained), 100)
    plt.plot(d_range_trained, np.exp(intercept_trained) * d_range_trained**slope_trained, 'r:', alpha=0.5, label=f'Fit Trained: b={slope_trained:.2f}')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Dimension d')
    plt.ylabel('Time (s) for Forward log-det (Batch=1000)')
    plt.title('Log-Det Computation Time vs Dimension')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    plt.savefig('benchmark_logdet_complexity.png', dpi=300, bbox_inches='tight')
    print("Saved benchmark_logdet_complexity.png")
    
if __name__ == "__main__":
    main()
