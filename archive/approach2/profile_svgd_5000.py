import torch
import time
import psutil
import os
import threading

torch.set_num_threads(2)

from config import set_seed
from svgd import svgd_sample
from oracle_mimo import solve_lambda_star_mimo_radial, log_w_J_mimo
from diagnostics import compute_mmd_rbf, compute_energy_distance

def monitor_cpu(stop_event, cpu_loads):
    while not stop_event.is_set():
        cpu_loads.append(psutil.cpu_percent(interval=0.5))

def run_profile():
    print("Starting profiling for nt=2, N=5000...")
    n_t = 2
    d = 4
    N = 5000
    n_steps = 500
    sigma2 = 0.1
    A = 10.0
    P = A**2 / 9.0
    
    mcmc_path = f"mcmc_reference_nt{n_t}.pt"
    ref_samples_full = torch.load(mcmc_path, weights_only=True)
    if ref_samples_full.shape[0] > N:
        ref_samples = ref_samples_full[:N]
    else:
        ref_samples = ref_samples_full
        
    lambda_star = solve_lambda_star_mimo_radial(P, A, n_t, sigma2)
    def log_w_J_wrapper(theta):
        return log_w_J_mimo(theta, lambda_star, sigma2, n_t)
        
    set_seed(42)
    
    stop_event = threading.Event()
    cpu_loads = []
    monitor_thread = threading.Thread(target=monitor_cpu, args=(stop_event, cpu_loads))
    monitor_thread.start()
    
    t0 = time.time()
    
    print("Running SVGD loop...")
    samples = svgd_sample(log_w_J_wrapper, N, n_steps, dim=d, lr=0.5)
    t_svgd = time.time() - t0
    
    print("Computing metrics...")
    t1 = time.time()
    compute_mmd_rbf(samples, ref_samples, bandwidth='median')
    compute_energy_distance(samples, ref_samples)
    t_metrics = time.time() - t1
    
    stop_event.set()
    monitor_thread.join()
    
    avg_cpu = sum(cpu_loads) / len(cpu_loads) if cpu_loads else 0.0
    max_cpu = max(cpu_loads) if cpu_loads else 0.0
    total_time = t_svgd + t_metrics
    
    print(f"\n--- Profiling Results ---")
    print(f"Time for SVGD ({n_steps} steps): {t_svgd:.1f}s")
    print(f"Time for Metrics (MMD+ED): {t_metrics:.1f}s")
    print(f"Total time for 1 seed (nt={n_t}, N={N}): {total_time:.1f}s")
    print(f"Estimated total time (3 dim * 5 seeds) = {total_time * 15 / 60:.1f} minutes")
    print(f"Average CPU Load: {avg_cpu:.1f}%")
    print(f"Max CPU Load: {max_cpu:.1f}%")

if __name__ == "__main__":
    run_profile()
