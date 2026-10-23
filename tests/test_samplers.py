import torch

from amortized_mcmc import run_amortized_mh
from flow import RealNVPFlow
from svgd import svgd_update


def test_independent_mh_accepts_everything_when_proposal_equals_target():
    torch.manual_seed(0)
    flow = RealNVPFlow(dim=4, n_coupling_layers=2, hidden_dim=8)
    with torch.no_grad():
        target = lambda theta: flow.log_prob(theta)
        samples, accept_rate, rhat = run_amortized_mh(target, flow, n_chains=4, n_samples=200, n_warmup=10)
    assert samples.shape == (200, 4, 4)
    assert accept_rate > 0.999


def test_svgd_update_shape_and_finiteness():
    torch.manual_seed(0)
    X = torch.randn(50, 3, requires_grad=True)
    update = svgd_update(X, lambda x: -0.5 * (x ** 2).sum(-1))
    assert update.shape == (50, 3)
    assert torch.isfinite(update).all()
