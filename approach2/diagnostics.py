import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Optional

def compute_ess(theta: torch.Tensor, log_w_J_values: torch.Tensor, log_q_values: torch.Tensor) -> Tuple[float, float]:
    """
    Compute Effective Sample Size (ESS).
    ESS = 1 / sum(w_normalized**2)
    """
    log_weights = log_w_J_values.squeeze() - log_q_values.squeeze()
    log_weights = log_weights - torch.max(log_weights)
    
    w_unnorm = torch.exp(log_weights)
    w_norm = w_unnorm / torch.sum(w_unnorm)
    
    ess = 1.0 / torch.sum(w_norm ** 2).item()
    N = theta.shape[0]
    return ess, ess / N

def compute_mmd_rbf(x: torch.Tensor, y: torch.Tensor, bandwidth='median') -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) with RBF kernel.
    Following Gretton et al. (2012).
    """
    xx = torch.cdist(x, x, p=2.0)**2
    yy = torch.cdist(y, y, p=2.0)**2
    xy = torch.cdist(x, y, p=2.0)**2
    
    if bandwidth == 'median':
        # Median heuristic
        # Combine all pairwise distances (or just sample a subset to be fast)
        dists = torch.cat([xx.flatten(), yy.flatten(), xy.flatten()])
        # to save time, we can subsample if it's too large
        if dists.numel() > 100000:
            indices = torch.randperm(dists.numel())[:100000]
            dists = dists[indices]
        bandwidth = torch.median(torch.sqrt(dists)).item()
        # prevent divide by zero
        if bandwidth < 1e-5:
            bandwidth = 1.0
            
    k_xx = torch.exp(-xx / (2 * bandwidth**2)).mean()
    k_yy = torch.exp(-yy / (2 * bandwidth**2)).mean()
    k_xy = torch.exp(-xy / (2 * bandwidth**2)).mean()
    
    return (k_xx + k_yy - 2 * k_xy).item()

def compute_energy_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Energy Distance.
    Following Székely & Rizzo (2013).
    """
    xx = torch.cdist(x, x, p=2.0).mean()
    yy = torch.cdist(y, y, p=2.0).mean()
    xy = torch.cdist(x, y, p=2.0).mean()
    
    return (2 * xy - xx - yy).item()

def sir_resample(theta: torch.Tensor, log_w_J_values: torch.Tensor, log_q_values: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Sampling Importance Resampling (SIR) to generate reference samples from the target w_J
    using the flow as a proposal.
    """
    log_weights = log_w_J_values.squeeze() - log_q_values.squeeze()
    log_weights = log_weights - torch.max(log_weights)
    
    w_unnorm = torch.exp(log_weights)
    w_norm = w_unnorm / torch.sum(w_unnorm)
    
    indices = torch.multinomial(w_norm, n_samples, replacement=True)
    return theta[indices]

def compare_to_theoretical_1d(samples: torch.Tensor, theoretical_pdf_fn: Callable, save_path: str):
    """
    Histogram of samples vs theoretical PDF.
    """
    samples_np = samples.detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(8, 5))
    plt.hist(samples_np, bins=100, density=True, alpha=0.6, label="Flow Samples")
    
    x_min, x_max = samples_np.min() - 1.0, samples_np.max() + 1.0
    grid = torch.linspace(x_min, x_max, 500)
    pdf_vals = theoretical_pdf_fn(grid)
    
    plt.plot(grid.numpy(), pdf_vals.numpy(), label="Theoretical PDF", linewidth=2, color="red")
    
    plt.title("Flow Samples vs Theoretical PDF (1D)")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def compare_2d(samples: torch.Tensor, theoretical_pdf_fn_or_None: Optional[Callable], save_path: str):
    """
    Scatter plot of 2D samples.
    """
    samples_np = samples.detach().cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.3, s=2, label="Samples")
    
    if theoretical_pdf_fn_or_None is not None:
        pass # Add contours if a function is provided
        
    plt.title("Flow Samples (2D)")
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.savefig(save_path)
    plt.close()
