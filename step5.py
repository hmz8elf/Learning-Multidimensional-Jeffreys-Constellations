import time
import numpy as np
import matplotlib.pyplot as plt
from jeffreys_baseline import BaselineConfig, JeffreysConstellationBuilder
from validate_step4 import make_csi_model

def measure_time(d, M, markov_order):
    # Pour le cas CSI, la dimension d correspond à 2 * nt
    # On gère si d est impair (bien que normalement pair)
    nt = max(1, d // 2)
    sigma2 = 0.1
    A = 10.0
    P = A**2 / 9.0
    p = 6
    n_quad_marg = 8
    n_quad_fit = 40
    
    J_func, c_func, _ = make_csi_model(nt, sigma2)
    config = BaselineConfig(A=A, d=d, M=M, p=p,
                            n_quad_marg=n_quad_marg, n_quad_fit=n_quad_fit,
                            domain="ball", markov_order=markov_order)
    builder = JeffreysConstellationBuilder(J_func=J_func, c_func=c_func, P=P, config=config)
    
    t0 = time.perf_counter()
    # On mesure le temps total
    lam = builder.compute_lambda_star(verbose=False)
    X = builder.build(verbose=False)
    t1 = time.perf_counter()
    return t1 - t0

def run_benchmark():
    print("======================================================================")
    print("Étape 5 — Benchmark de complexité (cas III-D CSI imparfait)")
    print("======================================================================\n")

    # 1. Fix M, vary d
    ds = range(2, 10, 2)
    M_fixed = 4
    times_d = []
    print(f"--- 1. Variation de la dimension d (M fixé à {M_fixed}) ---")
    for d in ds:
        t = measure_time(d, M_fixed, markov_order=None)
        print(f"d={d} | M={M_fixed} | Temps = {t:.2f} s")
        times_d.append(t)

    # 2. Fix d, vary M
    Ms = range(2, 20, 2)
    d_fixed = 4
    times_M = []
    print(f"\n--- 2. Variation de M (dimension fixée à d={d_fixed}) ---")
    for M in Ms:
        t = measure_time(d_fixed, M, markov_order=None)
        print(f"d={d_fixed} | M={M} | Temps = {t:.2f} s")
        times_M.append(t)

    # 3. Vary markov_order on curves (Fix M, vary d)
    markovs = [None, 0, 1, 2, 3] # Tu peux maintenant ajouter d'autres ordres librement
    ds_markov = range(3, 11, 2)
    M_markov = 3
    times_markov = {mo: [] for mo in markovs}
    print(f"\n--- 3. Impact de markov_order (Courbes en fonction de d, M={M_markov}) ---")
    for mo in markovs:
        print(f" -> markov_order = {mo}")
        for d in ds_markov:
            t = measure_time(d, M_markov, markov_order=mo)
            print(f"    d={d} | M={M_markov} | Temps = {t:.2f} s")
            times_markov[mo].append(t)
        
    # --- Tracé des graphiques ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Graphique 1: vs d
    axs[0].plot(ds, times_d, marker='o', linestyle='-', color='b')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Dimension d')
    axs[0].set_ylabel('Temps (secondes)')
    axs[0].set_title(f'Croissance en d (M={M_fixed})')
    axs[0].set_xticks(ds)
    axs[0].grid(True, which="both", ls="--")
    
    # Graphique 2: vs M (log-log)
    axs[1].plot(Ms, times_M, marker='o', linestyle='-', color='g')
    axs[1].set_yscale('log')
    axs[1].set_xscale('log')
    axs[1].set_xlabel('M points par coordonnée')
    axs[1].set_title(f'Croissance en M (d={d_fixed})')
    axs[1].set_xticks(Ms)
    axs[1].set_xticklabels([str(m) for m in Ms])
    axs[1].grid(True, which="both", ls="--")
    
    # Graphique 3: markov_order
    # Définition d'une palette étendue au lieu d'un dictionnaire figé
    color_palette = ['red', 'green', 'orange', 'purple', 'cyan', 'brown', 'magenta', 'olive']
    
    for i, mo in enumerate(markovs):
        # Génération dynamique des labels
        if mo is None:
            lbl = 'Complet (None)'
        elif mo == 0:
            lbl = 'Séparable (Markov-0)'
        else:
            lbl = f'Markov-{mo}'
            
        # Attribution cyclique de la couleur
        c = color_palette[i % len(color_palette)]
        
        axs[2].plot(ds_markov, times_markov[mo], marker='s', linestyle='-', color=c, label=lbl)
        
    axs[2].set_yscale('log')
    axs[2].set_xlabel('Dimension d')
    axs[2].set_title(f'Impact du markov_order (M={M_markov})')
    axs[2].set_xticks(ds_markov)
    axs[2].legend()
    axs[2].grid(True, which="both", ls="--")
    
    plt.tight_layout()
    plot_path = "step5_complexity.png"
    plt.savefig(plot_path)
    print(f"\n==> Graphiques sauvegardés dans {plot_path}")

if __name__ == '__main__':
    run_benchmark()