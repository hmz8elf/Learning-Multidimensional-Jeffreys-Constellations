import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from jeffreys_baseline import BaselineConfig, JeffreysConstellationBuilder
from validate_step2 import adc_case
from blahut_arimoto import blahut_arimoto, mutual_information_uniform

Q = norm.sf

def compute_adc_transition_matrix(X, A, L):
    """
    Construit la matrice de transition P(Y|X) pour un ADC à L niveaux.
    X: Points de la constellation (shape: M)
    """
    X = np.asarray(X).flatten()
    M = len(X)
    if L == 2:
        thresholds = np.array([0.0])
    else:
        thresholds = np.linspace(-A, A, L - 1)
    
    t = np.concatenate([[-np.inf], thresholds, [np.inf]])
    W = np.zeros((M, L))
    for i in range(M):
        for j in range(L):
            t_lo, t_hi = t[j], t[j+1]
            prob_lo = Q(t_lo - X[i]) if np.isfinite(t_lo) else 1.0
            prob_hi = Q(t_hi - X[i]) if np.isfinite(t_hi) else 0.0
            W[i, j] = prob_lo - prob_hi
            
    # Sécurité numérique
    W = np.maximum(W, 1e-12)
    W = W / np.sum(W, axis=1, keepdims=True)
    return W

def get_pam_constellation(M, P):
    """
    Constellation PAM standard de taille M et puissance P.
    """
    if M == 1:
        return np.array([0.0])
    pam = np.arange(-(M-1), M+1, 2)
    current_power = np.mean(pam**2)
    return pam * np.sqrt(P / current_power)

def run_step6():
    print("======================================================================")
    print("Étape 6 — Information Mutuelle Uniforme (Cas III-B : ADC)")
    print("======================================================================\n")
    
    L = 8
    A = 5.0
    P_ratio = 1.0 / 9.0
    P = A**2 * P_ratio
    
    J_func, c_func, _ = adc_case(L, A)
    
    # Échelle de M allant jusqu'à 100 pour bien voir l'asymptote
    Ms = np.arange(2, 102, 4)
    im_approach1 = []
    im_pam = []
    
    # Capacité maximale théorique du canal AWGN sans ADC
    # P = 2.78, sigma^2 = 1
    C_awgn = 0.5 * np.log2(1 + P)
    
    print(f"Évaluation sur le cas ADC (L={L} niveaux, P={P:.2f})")
    print(f"Capacité AWGN théorique à P={P:.2f} : {C_awgn:.4f} bits")
    print(f"{'M':>4} | {'Approach I (bits)':>18} | {'PAM (bits)':>12}")
    print("-" * 40)
    
    for M in Ms:
        # --- Approach I ---
        config = BaselineConfig(A=A, d=1, M=M, p=6,
                                n_quad_marg=8, n_quad_fit=60)
        builder = JeffreysConstellationBuilder(
            J_func=lambda th: np.array([[J_func(th[0])]]),
            c_func=lambda th: c_func(th[0]),
            P=P, config=config)
        
        builder.compute_lambda_star(verbose=False)
        X_app1 = builder.build(verbose=False).flatten()
        
        W_app1 = compute_adc_transition_matrix(X_app1, A, L)
        # On utilise l'IM Uniforme pour respecter la contrainte de puissance P
        cap_app1 = mutual_information_uniform(W_app1)
        im_approach1.append(cap_app1)
        
        # --- PAM ---
        X_pam = get_pam_constellation(M, P)
        W_pam = compute_adc_transition_matrix(X_pam, A, L)
        cap_pam = mutual_information_uniform(W_pam)
        im_pam.append(cap_pam)
        
        print(f"{M:4d} | {cap_app1:18.4f} | {cap_pam:12.4f}")
        
    # --- Plot ---
    plt.figure(figsize=(8, 6))
    plt.plot(Ms, im_approach1, marker='o', linestyle='-', color='b', label='Approach I (Uniforme)')
    plt.plot(Ms, im_pam, marker='s', linestyle='--', color='r', label='Standard PAM (Uniforme)')
    
    # Ligne pour la capacité AWGN
    plt.axhline(y=C_awgn, color='g', linestyle=':', label=f'Capacité AWGN (C={C_awgn:.2f} bits)')
    
    plt.title(f'Information Mutuelle Uniforme vs Taille (ADC L={L}, P={P:.2f})')
    plt.xlabel('Taille de la constellation (M)')
    plt.ylabel('Information Mutuelle (bits/usage)')
    plt.legend()
    plt.grid(True, ls="--")
    
    plot_path = "step6_mutual_information.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\n==> Graphique sauvegardé dans {plot_path}")

if __name__ == '__main__':
    run_step6()
