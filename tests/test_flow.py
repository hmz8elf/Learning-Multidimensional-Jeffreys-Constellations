import math

import torch

from flow import RealNVPFlow


def _random_flow(dim, seed=0):
    """RealNVP with randomised weights (the default init is the identity map)."""
    torch.manual_seed(seed)
    flow = RealNVPFlow(dim=dim, n_coupling_layers=4, hidden_dim=16).double()
    with torch.no_grad():
        for p in flow.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return flow


def test_sample_shape():
    flow = _random_flow(4).float()  # sample() draws float32 base noise
    assert flow.sample(10).shape == (10, 4)


def test_forward_inverse_roundtrip():
    flow = _random_flow(6)
    z = torch.randn(100, 6, dtype=torch.float64)
    theta, logdet_fwd = flow.forward_full(z)
    z_back, logdet_inv = flow.inverse_full(theta)
    assert torch.allclose(z, z_back, atol=1e-10)
    assert torch.allclose(logdet_fwd, -logdet_inv, atol=1e-10)


def test_log_prob_matches_change_of_variables():
    # log q(theta) = log N(z; 0, I) - log|det d theta / d z|, Jacobian computed by autograd.
    dim = 4
    flow = _random_flow(dim, seed=1)
    z = torch.randn(5, dim, dtype=torch.float64)
    theta, _ = flow.forward_full(z)
    for i in range(z.shape[0]):
        jac = torch.autograd.functional.jacobian(lambda x: flow.forward_full(x[None])[0][0], z[i])
        log_base = -0.5 * (z[i] ** 2).sum() - 0.5 * dim * math.log(2 * math.pi)
        expected = log_base - torch.linalg.slogdet(jac)[1]
        got = flow.log_prob(theta[i:i + 1].detach())[0]
        assert torch.allclose(got, expected, atol=1e-8)
