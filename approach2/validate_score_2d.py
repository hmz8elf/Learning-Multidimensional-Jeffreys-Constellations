import torch
from config import set_seed
from score_based import score_based_sample
from oracle import log_w_J_2d, solve_lambda_star
import matplotlib.pyplot as plt

def validate_2d():
    print("Validating Score-based 2D...")
    set_seed(42)
    
    A = 10.0
    P = A**2 / 9
    lambda_star = solve_lambda_star(P, A)
    
    def log_w_J_wrapper(theta):
        return log_w_J_2d(theta, lambda_star)
        
    n_samples = 2000
    print(f"Running Score-based sampling on {n_samples} particles (2D)...")
    theta = score_based_sample(log_w_J_wrapper, n_samples, dim=2, 
                               L=10, T=100, sigma_max=2.0, sigma_min=0.01, epsilon_base=5e-3)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(theta[:, 0].numpy(), theta[:, 1].numpy(), s=1, alpha=0.5)
    plt.title("Score-based samples (2D)")
    plt.axis("equal")
    plt.savefig("score_vs_theoretical_2d.png")
    print("Saved plot to score_vs_theoretical_2d.png")
    print("PASS")

if __name__ == "__main__":
    validate_2d()
