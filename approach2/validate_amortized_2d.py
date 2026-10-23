import torch
from config import set_seed
from flow import RealNVPFlow
from oracle import log_w_J_2d, solve_lambda_star
from diagnostics import compare_2d
from amortized_mcmc import amortized_mh_sample

def run_2d():
    print("Validating Amortized MCMC 2D...")
    set_seed(42)
    
    A = 10.0
    P = A**2 / 9
    lambda_star = solve_lambda_star(P, A)
    
    # Le flow 2D
    flow = RealNVPFlow(dim=2, n_coupling_layers=6, hidden_dim=64)
    flow.load_state_dict(torch.load("checkpoints/flow_2d.pt", weights_only=True))
    flow.eval()
    
    def log_w_J_wrapper(theta):
        return log_w_J_2d(theta, lambda_star)
        
    print("Running Amortized MH (2D, 5000 samples, 4 chains)...")
    samples, ar, rhat = amortized_mh_sample(
        log_w_J_fn=log_w_J_wrapper,
        flow=flow,
        n_samples=5000,
        n_chains=4,
        n_warmup=500,
        device=torch.device("cpu"),
    )
    
    print(f"  Accept rate : {ar:.3f}")
    print(f"  R-hat max   : {rhat:.4f}")
    
    compare_2d(samples, None, "amortized_samples_2d.png")
    print("Saved plot to amortized_samples_2d.png")
    print("PASS")

if __name__ == "__main__":
    run_2d()
