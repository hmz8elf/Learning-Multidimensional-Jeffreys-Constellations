import torch
import numpy as np
import matplotlib.pyplot as plt
from oracle import log_w_J_1d, theoretical_jeffreys_pdf_1d, solve_lambda_star

def test_oracle():
    print("Testing oracle...")
    
    A = 10.0
    P = A**2 / 9
    lambda_star = solve_lambda_star(P, A)
    print(f"Calculated lambda_star: {lambda_star:.4f}")
    
    grid = torch.linspace(-A, A, 500)
    
    # Verify no NaN/Inf
    log_w = log_w_J_1d(grid, lambda_star)
    if torch.isnan(log_w).any() or torch.isinf(log_w).any():
        print("FAIL: NaN or Inf found in log_w_J_1d")
        return False
        
    # Manual normalization using trapezoidal rule
    unnorm_pdf = torch.exp(log_w)
    dx = grid[1] - grid[0]
    Z_manual = torch.trapz(unnorm_pdf, dx=dx)
    manual_normalized_pdf = unnorm_pdf / Z_manual
    
    # Theoretical PDF
    theoretical_pdf = theoretical_jeffreys_pdf_1d(grid, lambda_star)
    
    # Relative error
    # Avoid division by zero by adding small epsilon
    error = torch.abs(manual_normalized_pdf - theoretical_pdf) / (theoretical_pdf + 1e-8)
    max_error = torch.max(error).item()
    mean_error = torch.mean(error).item()
    
    print(f"Max relative error: {max_error:.6f}")
    print(f"Mean relative error: {mean_error:.6f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(grid.numpy(), theoretical_pdf.numpy(), label="Theoretical PDF", linestyle='dashed', linewidth=3)
    plt.plot(grid.numpy(), manual_normalized_pdf.numpy(), label="Manual Normalized PDF", alpha=0.7)
    plt.title("Oracle 1D vs Theoretical PDF")
    plt.xlabel(r"$\theta$")
    plt.ylabel("PDF")
    plt.legend()
    plt.grid(True)
    plt.savefig("oracle_vs_theoretical_1d.png")
    plt.close()
    print("Saved plot to oracle_vs_theoretical_1d.png")
    
    if max_error < 1e-2:
        print("PASS")
        return True
    else:
        print("FAIL: Relative error > 1e-2")
        return False

if __name__ == "__main__":
    test_oracle()
