import torch
import numpy as np
from config import FlowConfig, set_seed
from flow import RealNVPFlow
from train import train
from diagnostics import compute_ess, compare_2d
from oracle_mimo import solve_lambda_star_mimo_radial, log_w_J_mimo

def main():
    n_t = 4
    d = 2 * n_t
    sigma2 = 0.1
    A = 10.0
    P = A**2 / 9.0
    
    config = FlowConfig(dim=d, n_coupling_layers=10, hidden_dim=128, lr=5e-4, batch_size=1024, n_iters=4000, seed=42)
    set_seed(config.seed)
    
    flow = RealNVPFlow(dim=config.dim, n_coupling_layers=config.n_coupling_layers, hidden_dim=config.hidden_dim)
    
    lambda_star = solve_lambda_star_mimo_radial(P, A, n_t, sigma2)
    print(f"Using lambda_star: {lambda_star:.4f}")
    
    def log_w_J_wrapper(theta):
        return log_w_J_mimo(theta, lambda_star, sigma2, n_t)
        
    print(f"Training MIMO Flow (n_t={n_t}, d={d})...")
    flow, history = train(flow, log_w_J_wrapper, config, tag=f"mimo_nt{n_t}")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    
    print("Evaluating ESS...")
    flow.eval()
    with torch.no_grad():
        samples = flow.sample(5000)
        log_q = flow.log_prob(samples)
        log_w = log_w_J_wrapper(samples)
        ess, ess_n = compute_ess(samples, log_w, log_q)
        
    print(f"ESS: {ess:.2f}, ESS/N: {ess_n:.4f}")
    
if __name__ == "__main__":
    main()
