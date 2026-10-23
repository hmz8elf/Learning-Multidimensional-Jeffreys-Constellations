"""
Interface générique du Normalizing Flow.
Entrées/sorties stables pour réutilisation (SVGD / score-based sampling).
"""

import torch
import torch.nn as nn
from typing import Tuple

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        # Initialize output to zero so the coupling layer starts as an identity function
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CouplingLayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, layer_idx: int):
        super().__init__()
        self.dim = dim
        self.is_1d = (dim == 1)
        if self.is_1d:
            # 1D fallback: simple affine transformation.
            self.s = nn.Parameter(torch.zeros(1))
            self.t = nn.Parameter(torch.zeros(1))
        else:
            self.mask = torch.zeros(dim)
            if layer_idx % 2 == 0:
                self.mask[::2] = 1.0
            else:
                self.mask[1::2] = 1.0
            self.register_buffer("mask_buf", self.mask)
            
            in_dim = int(self.mask.sum().item())
            out_dim = dim - in_dim
            
            self.net_s = MLP(in_dim, out_dim, hidden_dim)
            self.net_t = MLP(in_dim, out_dim, hidden_dim)

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_1d:
            s = self.s.expand_as(theta)
            t = self.t.expand_as(theta)
            theta_out = theta * torch.exp(s) + t
            log_det = s.sum(dim=-1)
            return theta_out, log_det
        else:
            theta_A = theta * self.mask_buf
            theta_A_features = theta[:, self.mask_buf == 1.0]
            
            s = self.net_s(theta_A_features)
            t = self.net_t(theta_A_features)
            
            s_full = torch.zeros_like(theta)
            t_full = torch.zeros_like(theta)
            s_full[:, self.mask_buf == 0.0] = s
            t_full[:, self.mask_buf == 0.0] = t
            
            theta_out = theta_A + (theta * (1 - self.mask_buf)) * torch.exp(s_full) + t_full
            log_det = s_full.sum(dim=-1)
            return theta_out, log_det

    def inverse(self, theta_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_1d:
            s = self.s.expand_as(theta_out)
            t = self.t.expand_as(theta_out)
            theta = (theta_out - t) * torch.exp(-s)
            log_det = -s.sum(dim=-1)
            return theta, log_det
        else:
            theta_A = theta_out * self.mask_buf
            theta_A_features = theta_out[:, self.mask_buf == 1.0]
            
            s = self.net_s(theta_A_features)
            t = self.net_t(theta_A_features)
            
            s_full = torch.zeros_like(theta_out)
            t_full = torch.zeros_like(theta_out)
            s_full[:, self.mask_buf == 0.0] = s
            t_full[:, self.mask_buf == 0.0] = t
            
            theta = theta_A + (theta_out * (1 - self.mask_buf) - t_full) * torch.exp(-s_full)
            log_det = -s_full.sum(dim=-1)
            return theta, log_det


class RealNVPFlow(nn.Module):
    def __init__(self, dim: int, n_coupling_layers: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            CouplingLayer(dim, hidden_dim, i) for i in range(n_coupling_layers)
        ])
        
    def sample(self, n: int) -> torch.Tensor:
        device = next(self.parameters()).device
        z = torch.randn(n, self.dim, device=device)
        theta = z
        for layer in self.layers:
            theta, _ = layer.forward(theta)
        return theta

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        z = theta
        log_det_total = torch.zeros(theta.shape[0], device=theta.device)
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            log_det_total += log_det
            
        log_q0_z = -0.5 * (z ** 2 + 1.8378770664093453).sum(dim=-1)
        
        return log_q0_z + log_det_total

    def forward_full(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = z
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        for layer in self.layers:
            theta, log_det = layer.forward(theta)
            log_det_total += log_det
        return theta, log_det_total

    def inverse_full(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = theta
        log_det_total = torch.zeros(theta.shape[0], device=theta.device)
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            log_det_total += log_det
        return z, log_det_total
