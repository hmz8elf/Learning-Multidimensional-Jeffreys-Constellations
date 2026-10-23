"""Test pilote : mesure du temps réel pour 1 graine, nt=8 (pire cas) sur GPU."""
import torch, time
torch.set_num_threads(4)

from amortized_mcmc import load_flow, amortized_mh_sample
from oracle_mimo import solve_lambda_star_mimo_radial, log_w_J_mimo

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PILOT] Device = {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[PILOT] GPU = {torch.cuda.get_device_name(0)}")
    print(f"[PILOT] VRAM libre = {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

n_t = 8
A, P, SIGMA2 = 10.0, 10.0**2/9.0, 0.1
lambda_star = solve_lambda_star_mimo_radial(P, A, n_t, SIGMA2)
flow = load_flow(n_t, DEVICE)

def log_w_J_fn(theta):
    return log_w_J_mimo(theta, lambda_star, SIGMA2, n_t)

t0 = time.time()
samples, ar, rhat = amortized_mh_sample(
    log_w_J_fn, flow, n_samples=5000, n_chains=4, n_warmup=500, device=DEVICE
)
elapsed = time.time() - t0
print(f"[PILOT] 1 graine nt=8 : {elapsed:.2f}s")
print(f"[PILOT] Accept rate = {ar:.3f}, R-hat max = {rhat:.4f}")
print(f"[PILOT] Estimation totale (15 graines×dims) ≈ {elapsed*15:.1f}s = {elapsed*15/60:.1f} min")
