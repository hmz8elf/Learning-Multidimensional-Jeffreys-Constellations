import torch
from flow import RealNVPFlow

def test_invertibility():
    print("Testing invertibility...")
    
    # Test 1D and 2D
    for dim in [1, 2]:
        print(f"Dimension: {dim}")
        flow = RealNVPFlow(dim=dim, n_coupling_layers=4, hidden_dim=32)
        
        n = 100
        z = torch.randn(n, dim)
        
        # Forward pass
        theta, log_det_fwd = flow.forward_full(z)
        
        # Inverse pass
        z_rec, log_det_inv = flow.inverse_full(theta)
        
        # Check reconstruction
        error = torch.norm(z_rec - z, p=2, dim=-1).mean().item()
        print(f"Mean reconstruction error: {error:.2e}")
        
        if error > 1e-5:
            print("FAIL: Reconstruction error too high")
            return False
            
        # Check log_prob does not produce NaNs
        log_probs = flow.log_prob(flow.sample(n))
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            print("FAIL: NaN or Inf in log_prob")
            return False
            
    print("PASS")
    return True

if __name__ == "__main__":
    test_invertibility()
