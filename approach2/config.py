from dataclasses import dataclass
import torch
import numpy as np
import random

@dataclass
class FlowConfig:
    dim: int
    n_coupling_layers: int = 6
    hidden_dim: int = 64
    lr: float = 1e-3
    batch_size: int = 512
    n_iters: int = 5000
    seed: int = 0
    device: str = "cpu"

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
