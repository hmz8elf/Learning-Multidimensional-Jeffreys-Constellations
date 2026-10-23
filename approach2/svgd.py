import torch
import numpy as np
from typing import Callable

def svgd_update(X: torch.Tensor, log_w_J: Callable) -> torch.Tensor:
    """
    Calcule la direction de mise à jour SVGD pour les particules X.
    X : (n, d) avec requires_grad=True
    """
    n, d = X.shape
    
    # 1. Calcul du score (gradient de log w_J)
    log_p = log_w_J(X)
    
    # Gradient du log-prior (score)
    # On utilise sum() car les particules sont indépendantes
    score = torch.autograd.grad(log_p.sum(), X)[0]
    
    # 2. Calcul du noyau RBF
    # On détache X pour le calcul du noyau (SVGD classique)
    X_det = X.detach()
    diff = X_det.unsqueeze(1) - X_det.unsqueeze(0)  # (n, n, d)
    sq_dist = (diff ** 2).sum(dim=-1)               # (n, n)
    
    # Heuristique de la médiane
    # médian_dist = torch.median(sq_dist[sq_dist > 0]) si n est grand, mais on peut juste prendre la médiane
    med = torch.median(sq_dist)
    h = med / np.log(n) if med > 0 else torch.tensor(1.0, device=X.device)
    
    K = torch.exp(-sq_dist / h)  # (n, n)
    
    # 3. Calcul du terme répulsif (gradient du noyau)
    # sum_j K_ji = K.sum(dim=1)  car K est symétrique
    sum_K = K.sum(dim=1, keepdim=True)  # (n, 1)
    
    # repulsif = (2/h) * [ sum_K * X - K @ X ]
    repulsive = (2.0 / h) * (sum_K * X_det - torch.matmul(K, X_det))
    
    # 4. Update total
    # Phi(X) = (1/n) * sum_j [ K_ji * score_j + repulsif_i ]
    # K @ score calcule sum_j K_ij score_j
    update = (torch.matmul(K, score) + repulsive) / n
    
    return update


def svgd_sample(log_w_J: Callable, n_particles: int, n_steps: int, dim: int, 
                lr: float = 0.1, device: str = "cpu") -> torch.Tensor:
    """
    Génère des échantillons via SVGD.
    L'initialisation est faite aléatoirement (loi normale N(0, 1)) plutôt 
    qu'à partir du Flow. Cela permet d'évaluer SVGD comme une méthode 
    entièrement indépendante du Flow pour une comparaison équitable.
    """
    X = torch.randn(n_particles, dim, device=device)
    X.requires_grad_(True)
    
    optimizer = torch.optim.Adagrad([X], lr=lr)
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # Calcule l'update SVGD
        update = svgd_update(X, log_w_J)
        
        # On assigne -update au gradient pour que l'optimizer fasse: X = X + lr * update
        X.grad = -update
        
        optimizer.step()
        
        # Affiche la progression tous les 10%
        if (step + 1) % (n_steps // 10) == 0:
            print(f"SVGD Step {step+1}/{n_steps}, lr={lr}")
            
    return X.detach()
