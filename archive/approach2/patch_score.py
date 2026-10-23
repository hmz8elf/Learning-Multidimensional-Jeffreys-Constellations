import torch
import numpy as np

def score_based_sample(log_w_J, n_samples: int, dim: int, 
                       device: torch.device = torch.device('cpu'),
                       L: int = 10, T: int = 100, 
                       sigma_max: float = 2.0, sigma_min: float = 0.01,
                       epsilon_base: float = 1e-4) -> torch.Tensor:
    x = torch.randn(n_samples, dim, device=device) * sigma_max
    sigmas = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), L))
    
    for i, sigma in enumerate(sigmas):
        alpha_i = epsilon_base * (sigma / sigma_min)**2
        
        for t in range(T):
            x.requires_grad_(True)
            log_p = log_w_J(x)
            
            grad_log_p = torch.autograd.grad(log_p.sum(), x)[0]
            
            # Gradient clipping to prevent explosion
            grad_norm = grad_log_p.norm(dim=-1, keepdim=True)
            max_norm = 10.0
            grad_log_p = torch.where(grad_norm > max_norm, grad_log_p * (max_norm / grad_norm), grad_log_p)
            
            with torch.no_grad():
                z = torch.randn_like(x)
                x = x + (alpha_i / 2) * grad_log_p + np.sqrt(alpha_i) * z
            
            x = x.detach()
            
    return x

import score_based
score_based.score_based_sample = score_based_sample
with open('score_based.py', 'w') as f:
    f.write('''import torch
import numpy as np
from typing import Callable

def score_based_sample(log_w_J: Callable, n_samples: int, dim: int, 
                       device: torch.device = torch.device('cpu'),
                       L: int = 10, T: int = 100, 
                       sigma_max: float = 2.0, sigma_min: float = 0.01,
                       epsilon_base: float = 1e-4) -> torch.Tensor:
    x = torch.randn(n_samples, dim, device=device) * sigma_max
    sigmas = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), L))
    
    for i, sigma in enumerate(sigmas):
        alpha_i = epsilon_base * (sigma / sigma_min)**2
        for t in range(T):
            x.requires_grad_(True)
            log_p = log_w_J(x)
            grad_log_p = torch.autograd.grad(log_p.sum(), x)[0]
            
            grad_norm = grad_log_p.norm(dim=-1, keepdim=True)
            max_norm = 10.0
            grad_log_p = torch.where(grad_norm > max_norm, grad_log_p * (max_norm / grad_norm), grad_log_p)
            
            with torch.no_grad():
                z = torch.randn_like(x)
                x = x + (alpha_i / 2) * grad_log_p + np.sqrt(alpha_i) * z
            x = x.detach()
    return x
''')
