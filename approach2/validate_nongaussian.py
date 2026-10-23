import torch
from config import FlowConfig, set_seed
from flow import RealNVPFlow
from train import train
from diagnostics import compute_ess, compare_to_theoretical_1d
from oracle_nongaussian import solve_lambda_star_nongaussian, log_w_J_nongaussian_1d, theoretical_pdf_nongaussian_1d

def main():
    config = FlowConfig(dim=1, n_coupling_layers=6, hidden_dim=64, lr=1e-3, batch_size=1024, n_iters=3000, seed=42)
    set_seed(config.seed)
    
    flow = RealNVPFlow(dim=config.dim, n_coupling_layers=config.n_coupling_layers, hidden_dim=config.hidden_dim)
    
    A = 10.0
    P = A**2 / 9
    lambda_star = solve_lambda_star_nongaussian(P, A)
    print(f"Using lambda_star: {lambda_star:.4f}")
    
    def log_w_J_wrapper(theta):
        return log_w_J_nongaussian_1d(theta, lambda_star)
        
    print("Training 1D Non-Gaussian Flow...")
    flow, history = train(flow, log_w_J_wrapper, config, tag="1d_nongaussian")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    
    print("Evaluating ESS...")
    flow.eval()
    with torch.no_grad():
        samples = flow.sample(5000)
        log_q = flow.log_prob(samples)
        log_w = log_w_J_wrapper(samples)
        ess, ess_n = compute_ess(samples, log_w, log_q)
        
    print(f"ESS: {ess:.2f}, ESS/N: {ess_n:.4f}")
    
    # Save plot
    def theo_wrapper(x):
        return theoretical_pdf_nongaussian_1d(x, lambda_star)
    compare_to_theoretical_1d(samples, theo_wrapper, "validate_nongaussian.png")
    print("Saved plot to validate_nongaussian.png")
    
    if ess_n > 0.5:
        print("Status: PASS")
    else:
        print("Status: FAIL (ESS/N too low, but might be expected for non-gaussian with 1D realNVP)")

if __name__ == "__main__":
    main()
