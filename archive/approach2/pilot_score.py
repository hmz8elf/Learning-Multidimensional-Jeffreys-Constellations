import torch
import time
from score_based import score_based_sample
from oracle_mimo import solve_lambda_star_mimo_radial, log_w_J_mimo

def pilot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    n_samples = 5000
    L = 10
    T = 100
    
    nt = 8
    nr = 8
    d = 2 * nt
    sigma_n2 = 0.1
    P = float(nt)
    
    lambda_star = solve_lambda_star_mimo_radial(P, sigma_n2, nt, nr)
    
    def log_w_J_wrapper(theta):
        return log_w_J_mimo(theta, lambda_star, sigma_n2, nt)
        
    start_time = time.time()
    _ = score_based_sample(log_w_J_wrapper, n_samples=n_samples, dim=d, 
                           device=device, L=L, T=T)
    end_time = time.time()
    
    duration = end_time - start_time
    estimated_total = duration * 15
    print(f"Time for 1 seed at nt=8 (d=16): {duration:.2f} seconds")
    print(f"Estimated total time for all seeds and dimensions: {estimated_total:.2f} seconds")

if __name__ == "__main__":
    pilot()
