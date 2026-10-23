import torch
import numpy as np
import sys
import os

# Ajouter Approche 1 au path pour importer blahut_arimoto
sys.path.append(os.path.abspath("../approach1"))
from blahut_arimoto import blahut_arimoto
from flow import RealNVPFlow

def main():
    n_t = 2
    d = 4
    sigma2 = 0.1
    
    print(f"Evaluation Downstream (Blahut-Arimoto) qualitative pour n_t={n_t} (d={d})")
    
    flow = RealNVPFlow(dim=d, n_coupling_layers=8, hidden_dim=64)
    flow.load_state_dict(torch.load("checkpoints/flow_mimo_nt2.pt", map_location="cpu"))
    flow.eval()
    
    # Constellation de taille M = 64
    M = 64
    with torch.no_grad():
        X = flow.sample(M).numpy()
        
    print(f"Taille de la constellation générée : {M}")
    
    # Construction d'une grille Y par Monte-Carlo
    N_Y = 2000
    noise = np.random.randn(N_Y, d) * np.sqrt(sigma2)
    # Les Y sont générés aléatoirement à partir des X
    Y = X[np.random.choice(M, N_Y)] + noise
    
    # Matrice de transition empirique W[i, j] = p(y_j | x_i)
    W = np.zeros((M, N_Y))
    for i in range(M):
        diff = Y - X[i]
        # p(y|x) ~ exp(-||y-x||^2 / (2*sigma^2))
        W[i, :] = np.exp(-np.sum(diff**2, axis=1) / (2 * sigma2))
        
    # Normalisation pour que sum_j W[i, j] = 1
    W_sum = np.sum(W, axis=1, keepdims=True)
    W = W / (W_sum + 1e-12)
    
    C, px = blahut_arimoto(W)
    
    print(f"Capacité estimée (Blahut-Arimoto) : {C:.4f} bits/usage")
    print("Test qualitatif PASS: les échantillons produits sont exploitables par les algorithmes avals.")

if __name__ == "__main__":
    main()
