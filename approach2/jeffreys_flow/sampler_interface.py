from abc import ABC, abstractmethod
import torch
from typing import Optional

class Sampler(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        pass
        
    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """Sample n points. Returns Tensor of shape (n, dim)."""
        pass
        
    def log_prob(self, theta: torch.Tensor) -> Optional[torch.Tensor]:
        """Optional method for log density of the sampler."""
        return None
