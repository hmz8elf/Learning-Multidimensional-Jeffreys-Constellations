import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import RealNVPFlow as OriginalRealNVPFlow
from .sampler_interface import Sampler
from typing import Optional
import torch

class RealNVPFlowSampler(Sampler):
    def __init__(self, dim: int, n_coupling_layers: int = 6, hidden_dim: int = 64):
        self._dim = dim
        self.flow = OriginalRealNVPFlow(dim=dim, n_coupling_layers=n_coupling_layers, hidden_dim=hidden_dim)
        
    @property
    def dim(self) -> int:
        return self._dim
        
    def sample(self, n: int) -> torch.Tensor:
        return self.flow.sample(n)
        
    def log_prob(self, theta: torch.Tensor) -> Optional[torch.Tensor]:
        return self.flow.log_prob(theta)
