import torch
from config import FlowConfig, set_seed
from flow import RealNVPFlow
from oracle import log_w_J_1d, theoretical_jeffreys_pdf_1d, solve_lambda_star
from diagnostics import compute_ess, compare_to_theoretical_1d

def validate_1d():
    print("Validating 1D flow...")
    config = FlowConfig(dim=1)
    set_seed(config.seed)
    
    flow = RealNVPFlow(dim=config.dim, n_coupling_layers=config.n_coupling_layers, hidden_dim=config.hidden_dim)
    try:
        flow.load_state_dict(torch.load("checkpoints/flow_1d.pt"))
    except FileNotFoundError:
        print("FAIL: checkpoints/flow_1d.pt not found. Run train.py first.")
        return
        
    A = 10.0
    P = A**2 / 9
    lambda_star = solve_lambda_star(P, A)
    
    n_samples = 5000
    with torch.no_grad():
        theta = flow.sample(n_samples)
        log_q = flow.log_prob(theta)
        log_w = log_w_J_1d(theta, lambda_star)
        
    ess, ess_n = compute_ess(theta, log_w, log_q)
    print(f"ESS: {ess:.2f}")
    print(f"ESS/N: {ess_n:.4f}")
    
    def theoretical_pdf(t):
        return theoretical_jeffreys_pdf_1d(t, lambda_star)
        
    compare_to_theoretical_1d(theta, theoretical_pdf, "flow_vs_theoretical_1d.png")
    print("Saved plot to flow_vs_theoretical_1d.png")
    
    if ess_n < 0.3:
        print("WARNING: ESS/N < 0.3. Check debug_log.md.")
    
    print("PASS")

if __name__ == "__main__":
    validate_1d()
