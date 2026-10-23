"""
evaluate_svgd_metrics_gpu.py
============================
Évaluation SVGD avec protocole symétrique au Flow :
  - N = 5000 particules
  - 5 graines (42, 43, 44, 45, 46)
  - Dimensions nt = 2, 4, 8 traitées STRICTEMENT l'une après l'autre
  - Device : CUDA (RTX 4060 Laptop GPU)
  - torch.set_num_threads(4) (moitié des 10 cœurs CPU)
  - Sauvegarde incrémentale après chaque dimension
  - gc.collect() + torch.cuda.empty_cache() entre dimensions
"""

import torch
torch.set_num_threads(4)

import numpy as np
import json
import os
import gc
import time
import psutil
import threading

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] torch.get_num_threads() = {torch.get_num_threads()}")
print(f"[INFO] Device = {DEVICE}")
if DEVICE == "cuda":
    print(f"[INFO] GPU = {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM free = {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

from config import set_seed
from svgd import svgd_sample
from diagnostics import compute_mmd_rbf, compute_energy_distance
from oracle_mimo import solve_lambda_star_mimo_radial, log_w_J_mimo

# ── Configuration ──────────────────────────────────────────────────────────────
N_SAMPLES = 5000
N_STEPS   = 500
SEEDS     = [42, 43, 44, 45, 46]  # 5 graines, identique au Flow
CONFIGS   = [2, 4, 8]             # nt values
LR        = 0.5
SIGMA2    = 0.1
A         = 10.0
P         = A**2 / 9.0
OUTPUT    = "svgd_metrics.json"   # fichier final (remplace l'ancien)
OUTPUT_INC = "svgd_metrics_gpu_incremental.json"  # sauvegarde incrémentale

# ── CPU monitor ────────────────────────────────────────────────────────────────
cpu_log = []
stop_monitor = threading.Event()

def _monitor(interval=3.0):
    while not stop_monitor.is_set():
        cpu_log.append(psutil.cpu_percent(interval=interval))

monitor_thread = threading.Thread(target=_monitor, daemon=True)
monitor_thread.start()

# ── Helper : charger les résultats partiels si interruption ───────────────────
results = []
if os.path.exists(OUTPUT_INC):
    try:
        with open(OUTPUT_INC) as f:
            results = json.load(f)
        already_done = [r["n_t"] for r in results]
        print(f"[INFO] Résultats partiels trouvés pour nt = {already_done}")
    except Exception:
        results = []

already_done_nts = {r["n_t"] for r in results}

# ── Boucle principale ──────────────────────────────────────────────────────────
wall_start = time.time()

for n_t in CONFIGS:
    if n_t in already_done_nts:
        print(f"\n[SKIP] nt={n_t} déjà calculé, on passe.")
        continue

    d = 2 * n_t
    print(f"\n{'='*60}")
    print(f"[DIM] n_t = {n_t}  (d = {d})")
    print(f"{'='*60}")

    mcmc_path = f"mcmc_reference_nt{n_t}.pt"
    if not os.path.exists(mcmc_path):
        print(f"[ERROR] Référence MCMC introuvable : {mcmc_path}  — on saute.")
        continue

    ref_samples_full = torch.load(mcmc_path, weights_only=True).to(DEVICE)
    lambda_star = solve_lambda_star_mimo_radial(P, A, n_t, SIGMA2)

    def log_w_J_wrapper(theta):
        return log_w_J_mimo(theta, lambda_star, SIGMA2, n_t)

    mmds, energies = [], []
    dim_start = time.time()

    for seed in SEEDS:
        set_seed(seed)
        print(f"\n  [SEED {seed}] nt={n_t}, N={N_SAMPLES}, steps={N_STEPS} ...")
        t0 = time.time()

        samples = svgd_sample(log_w_J_wrapper, N_SAMPLES, N_STEPS,
                              dim=d, lr=LR, device=DEVICE)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t_svgd = time.time() - t0
        print(f"  [SEED {seed}] SVGD terminé en {t_svgd:.1f}s")

        # Sous-échantillonner la référence si plus grande
        if ref_samples_full.shape[0] > N_SAMPLES:
            idx = torch.randperm(ref_samples_full.shape[0], device=DEVICE)[:N_SAMPLES]
            ref_samples = ref_samples_full[idx]
        else:
            ref_samples = ref_samples_full

        mmd    = compute_mmd_rbf(samples, ref_samples, bandwidth='median')
        energy = compute_energy_distance(samples, ref_samples)
        print(f"  [SEED {seed}] MMD={mmd:.6f}  Energy={energy:.6f}")

        mmds.append(mmd)
        energies.append(energy)

        # Libération VRAM après chaque graine
        del samples
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    dim_time = time.time() - dim_start
    print(f"\n[DIM] nt={n_t} terminé en {dim_time:.1f}s ({dim_time/60:.1f} min)")
    print(f"[DIM] MMD    : {np.mean(mmds):.6f} ± {np.std(mmds):.6f}")
    print(f"[DIM] Energy : {np.mean(energies):.6f} ± {np.std(energies):.6f}")

    res = {
        "n_t": n_t,
        "d": d,
        "N": N_SAMPLES,
        "n_seeds": len(SEEDS),
        "ESS_N_mean": "N/A",
        "ESS_N_std":  "N/A",
        "MMD_mean":              float(np.mean(mmds)),
        "MMD_std":               float(np.std(mmds)),
        "Energy_Distance_mean":  float(np.mean(energies)),
        "Energy_Distance_std":   float(np.std(energies)),
        "wall_time_s": round(dim_time, 2),
    }
    results.append(res)

    # ── Sauvegarde incrémentale ─────────────────────────────────────────────
    with open(OUTPUT_INC, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[SAVE] Résultats intermédiaires sauvegardés → {OUTPUT_INC}")

    # ── Libération mémoire entre dimensions ─────────────────────────────────
    del ref_samples_full, ref_samples
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    if DEVICE == "cuda":
        vram_free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"[MEM] VRAM libre après nt={n_t} : {vram_free:.2f} GB")

    cpu_now = psutil.cpu_percent(interval=1)
    print(f"[MEM] CPU actuel après nt={n_t} : {cpu_now:.1f}%  (attendu ~< 10 % en idle)")

# ── Résultats finaux ───────────────────────────────────────────────────────────
stop_monitor.set()
monitor_thread.join(timeout=5)

total_time = time.time() - wall_start
print(f"\n{'='*60}")
print(f"[DONE] Temps total : {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"[DONE] CPU max observé : {max(cpu_log) if cpu_log else 'N/A'}%")
print(f"[DONE] CPU moyen      : {np.mean(cpu_log) if cpu_log else 'N/A':.1f}%")

# Sauvegarde finale (remplace svgd_metrics.json)
with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=4)
print(f"[SAVE] Résultats finaux sauvegardés → {OUTPUT}")

print("\n===== TABLEAU FINAL =====")
print(f"{'nt':>4} | {'MMD_mean':>12} | {'MMD_std':>10} | {'Energy_mean':>14} | {'Energy_std':>12}")
print("-" * 62)
for r in results:
    print(f"{r['n_t']:>4} | {r['MMD_mean']:>12.6f} | {r['MMD_std']:>10.6f} | "
          f"{r['Energy_Distance_mean']:>14.6f} | {r['Energy_Distance_std']:>12.6f}")
print("=========================")
print(f"Protocole : N={N_SAMPLES}, {len(SEEDS)} graines, device={DEVICE}")
