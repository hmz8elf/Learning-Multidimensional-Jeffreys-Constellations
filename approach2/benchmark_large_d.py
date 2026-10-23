import torch
import time
from flow import RealNVPFlow

device = "cuda" if torch.cuda.is_available() else "cpu"
for d in [64, 256, 1024, 4096, 16384]:
    flow = RealNVPFlow(dim=d, n_coupling_layers=8, hidden_dim=64).to(device)
    z = torch.randn(1000, d, device=device)
    for _ in range(10): flow.forward_full(z)
    start = time.perf_counter()
    for _ in range(20): flow.forward_full(z)
    end = time.perf_counter()
    print(f"d={d}: {(end-start)/20*1000:.3f} ms")
