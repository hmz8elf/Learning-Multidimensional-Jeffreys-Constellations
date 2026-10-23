import torch
from config import FlowConfig, set_seed
from flow import RealNVPFlow
from oracle import log_w_J_2d, solve_lambda_star
from diagnostics import compute_ess, compare_2d
from train import train

def run_2d():
    print("Training and Validating 2D flow...")
    config = FlowConfig(dim=2, n_iters=3000, lr=1e-3, batch_size=1024)
    set_seed(config.seed)
    
    flow = RealNVPFlow(dim=config.dim, n_coupling_layers=config.n_coupling_layers, hidden_dim=config.hidden_dim)
    
    A = 10.0
    P = A**2 / 9
    lambda_star = solve_lambda_star(P, A)
    
    def log_w_J_wrapper(theta):
        return log_w_J_2d(theta, lambda_star)
        
    print("Starting 2D Training...")
    flow, history = train(flow, log_w_J_wrapper, config, tag="2d")
    
    n_samples = 5000
    with torch.no_grad():
        theta = flow.sample(n_samples)
        log_q = flow.log_prob(theta)
        log_w = log_w_J_wrapper(theta)
        
    ess, ess_n = compute_ess(theta, log_w, log_q)
    print(f"ESS (2D): {ess:.2f}")
    print(f"ESS/N (2D): {ess_n:.4f}")
    
    compare_2d(theta, None, "flow_samples_2d.png")
    print("Saved plot to flow_samples_2d.png")
    
    if ess_n < 0.3:
        print("WARNING: ESS/N < 0.3 for 2D. Check debug_log.md.")
        
    print("PASS")

if __name__ == "__main__":
    run_2d()
