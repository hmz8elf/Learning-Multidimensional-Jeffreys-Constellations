import sys, os
sys.path.append(os.path.abspath("../Approche 1"))
from itertools import product
import numpy as np
import time
from blahut_arimoto import blahut_arimoto

SIGMA2 = 0.1
N_Y = 2000
BA_MAX_ITER = 500
BA_TOL = 1e-6
M = 256
d = 16

def build_transition_matrix(X, sigma2, N_Y, rng):
    M, d = X.shape
    idx   = rng.integers(0, M, size=N_Y)
    noise = rng.standard_normal((N_Y, d)) * np.sqrt(sigma2)
    Y     = X[idx] + noise
    diff    = Y[None, :, :] - X[:, None, :]
    sq_dist = np.sum(diff ** 2, axis=-1)
    W       = np.exp(-sq_dist / (2 * sigma2))
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-300, 1.0, row_sums)
    W /= row_sums
    return W

def compute_mi_ba(X, sigma2, N_Y, rng):
    W  = build_transition_matrix(X, sigma2, N_Y, rng)
    C, _ = blahut_arimoto(W, max_iter=BA_MAX_ITER, tol=BA_TOL)
    return float(C)

rng = np.random.default_rng(42)
X = rng.standard_normal((M, d))
t0 = time.time()
compute_mi_ba(X, SIGMA2, N_Y, rng)
t1 = time.time()
print(f"Time for 1 run (M=256, d=16): {t1-t0:.4f} seconds")

# Memory estimate
M_val = 256
N_Y_val = 2000
d_val = 16
W_bytes  = M_val * N_Y_val * 8
diff_bytes = M_val * N_Y_val * d_val * 8
total_MB = (W_bytes + diff_bytes) / 1e6
print(f"Memory for 1 run (M=256, d=16): {total_MB:.2f} MB")
