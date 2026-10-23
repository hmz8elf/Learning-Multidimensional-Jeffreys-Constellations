import torch
from config import FlowConfig, set_seed
from svgd import svgd_sample
from oracle import log_w_J_1d, theoretical_jeffreys_pdf_1d, solve_lambda_star
from diagnostics import compare_to_theoretical_1d

def validate_1d():
    print("Validating SVGD 1D...")
    set_seed(42)
    
    A = 10.0
    P = A**2 / 9
    lambda_star = solve_lambda_star(P, A)
    
    def log_w_J_wrapper(theta):
        return log_w_J_1d(theta, lambda_star)
        
    n_samples = 2000
    n_steps = 1000
    lr = 0.5
    
    print(f"Running SVGD for {n_steps} steps on {n_samples} particles...")
    theta = svgd_sample(log_w_J_wrapper, n_samples, n_steps, dim=1, lr=lr)
    
    def theoretical_pdf(t):
        return theoretical_jeffreys_pdf_1d(t, lambda_star)
        
    compare_to_theoretical_1d(theta, theoretical_pdf, "svgd_vs_theoretical_1d.png")
    print("Saved plot to svgd_vs_theoretical_1d.png")
    
    print("PASS")

if __name__ == "__main__":
    validate_1d()
