import torch
import json
import os
from flow import RealNVPFlow
from config import FlowConfig, set_seed
from typing import Callable, Tuple

def train(flow: RealNVPFlow, log_w_J: Callable, config: FlowConfig, tag: str) -> Tuple[RealNVPFlow, dict]:
    flow.to(config.device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=config.lr)
    
    history = {"loss": []}
    
    for i in range(config.n_iters):
        optimizer.zero_grad()
        
        # Reparameterization trick: gradients flow through sample()
        theta = flow.sample(config.batch_size)
        
        # Reverse KL Divergence objective
        log_q = flow.log_prob(theta)
        log_w = log_w_J(theta)
        
        loss = (log_q - log_w).mean()
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            loss_val = loss.item()
            history["loss"].append(loss_val)
            print(f"Iter {i+1}/{config.n_iters}, Loss: {loss_val:.4f}")
            
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(flow.state_dict(), f"checkpoints/flow_{tag}.pt")
    with open(f"checkpoints/history_{tag}.json", "w") as f:
        json.dump(history, f)
        
    return flow, history

if __name__ == "__main__":
    from oracle import log_w_J_1d, solve_lambda_star
    
    config = FlowConfig(dim=1, n_coupling_layers=6, hidden_dim=64, lr=1e-3, batch_size=1024, n_iters=3000, seed=42)
    set_seed(config.seed)
    
    flow = RealNVPFlow(dim=config.dim, n_coupling_layers=config.n_coupling_layers, hidden_dim=config.hidden_dim)
    
    A = 10.0
    P = A**2 / 9
    lambda_star = solve_lambda_star(P, A)
    print(f"Using lambda_star: {lambda_star:.4f}")
    
    def log_w_J_wrapper(theta):
        return log_w_J_1d(theta, lambda_star)
        
    print("Training 1D Flow...")
    flow, history = train(flow, log_w_J_wrapper, config, tag="1d")
    print(f"Final loss: {history['loss'][-1]:.4f}")
