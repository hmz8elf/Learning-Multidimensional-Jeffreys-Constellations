import torch
from config import set_seed
from score_based import score_based_sample
from oracle import log_w_J_1d, theoretical_jeffreys_pdf_1d, solve_lambda_star
from diagnostics import compare_to_theoretical_1d

def validate_1d():
    print("Validating Score-based 1D...")
    set_seed(42)
    
    A = 10.0
    P = A**2 / 9
    lambda_star = solve_lambda_star(P, A)
    
    def log_w_J_wrapper(theta):
        return log_w_J_1d(theta, lambda_star)
        
    n_samples = 2000
    
    print(f"Running Score-based sampling on {n_samples} particles...")
    theta = score_based_sample(log_w_J_wrapper, n_samples, dim=1, 
                               L=10, T=100, sigma_max=2.0, sigma_min=0.01, epsilon_base=5e-3)
    
    def theoretical_pdf(t):
        return theoretical_jeffreys_pdf_1d(t, lambda_star)
        
    compare_to_theoretical_1d(theta, theoretical_pdf, "score_vs_theoretical_1d.png")
    print("Saved plot to score_vs_theoretical_1d.png")
    
    print("PASS")

if __name__ == "__main__":
    validate_1d()
