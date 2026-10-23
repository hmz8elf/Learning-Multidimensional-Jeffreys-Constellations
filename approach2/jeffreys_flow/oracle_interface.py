from abc import ABC, abstractmethod
import torch

class Oracle(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @abstractmethod
    def log_w_J(self, theta: torch.Tensor) -> torch.Tensor:
        pass
