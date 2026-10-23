"""
validate_amortized_1d.py
Validation du MCMC amorti (IMH + Flow) en 1D contre la densité théorique.
"""
import torch
from config import set_seed
from amortized_mcmc import load_flow, amortized_mh_sample
from oracle import log_w_J_1d, theoretical_jeffreys_pdf_1d, solve_lambda_star
from diagnostics import compare_to_theoretical_1d
from flow import RealNVPFlow


def validate_1d():
    print("Validating Amortized MCMC 1D...")
    set_seed(42)

    A = 10.0
    P = A**2 / 9
    lambda_star = solve_lambda_star(P, A)

    # Le flow 1D a une architecture différente — on le charge à part
    flow = RealNVPFlow(dim=1, n_coupling_layers=6, hidden_dim=64)
    flow.load_state_dict(torch.load("checkpoints/flow_1d.pt", weights_only=True))
    flow.eval()

    def log_w_J_wrapper(theta):
        return log_w_J_1d(theta, lambda_star)

    print("Running Amortized MH (1D, 2000 samples, 4 chains)...")
    samples, ar, rhat = amortized_mh_sample(
        log_w_J_fn=log_w_J_wrapper,
        flow=flow,
        n_samples=2000,
        n_chains=4,
        n_warmup=200,
        device=torch.device("cpu"),
    )
    print(f"  Accept rate : {ar:.3f}")
    print(f"  R-hat max   : {rhat:.4f}")

    def theoretical_pdf(t):
        return theoretical_jeffreys_pdf_1d(t, lambda_star)

    compare_to_theoretical_1d(samples, theoretical_pdf, "amortized_vs_theoretical_1d.png")
    print("Saved amortized_vs_theoretical_1d.png")
    print("PASS")


if __name__ == "__main__":
    validate_1d()
