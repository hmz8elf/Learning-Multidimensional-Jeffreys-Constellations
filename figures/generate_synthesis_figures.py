import os
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def power_law(x, a, b):
    return a * np.power(x, b)

# Harmonized styling
COLORS = {
    'Flow': '#1f77b4',       # blue
    'SVGD': '#ff7f0e',       # orange
    'Score-based': '#2ca02c', # green
    'MCMC amorti': '#d62728', # red
    'PAM': '#7f7f7f',        # grey
    'Jeffreys 1D': '#9467bd' # purple
}
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'legend.fontsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11})

os.makedirs('figures', exist_ok=True)

# ---------------------------------------------------------
# Figure 1: Energy Distance vs Dimension d
# ---------------------------------------------------------
try:
    with open('approach2/evaluation_metrics.json', 'r') as f:
        flow_data = json.load(f)
    with open('approach2/svgd_metrics.json', 'r') as f:
        svgd_data = json.load(f)
    with open('approach2/score_metrics.json', 'r') as f:
        score_data = json.load(f)
    with open('approach2/amortized_metrics.json', 'r') as f:
        mcmc_data = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 6))

    def plot_method(data, name, label):
        d_vals = []
        ed_vals = []
        for item in data:
            if 'd' in item and 'Energy_Distance_mean' in item:
                d_vals.append(item['d'])
                ed_vals.append(item['Energy_Distance_mean'])
        
        # Sort by d
        sorted_indices = np.argsort(d_vals)
        d_vals = np.array(d_vals)[sorted_indices]
        ed_vals = np.array(ed_vals)[sorted_indices]
        
        ax.plot(d_vals, ed_vals, marker='o', linewidth=2, color=COLORS[name], label=label)

    plot_method(flow_data, 'Flow', 'Normalizing Flow')
    plot_method(svgd_data, 'SVGD', 'SVGD')
    plot_method(score_data, 'Score-based', 'Score-based Generative Model')
    plot_method(mcmc_data, 'MCMC amorti', 'MCMC Amorti')

    ax.set_xlabel('Dimension $d = 2 n_t$')
    ax.set_ylabel('Energy Distance (moyenne)')
    ax.set_yscale('log')
    ax.set_xticks([4, 8, 16])
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    ax.set_title('Distance de l\'Énergie entre les échantillons et la cible analytique vs Dimension')
    plt.tight_layout()
    plt.savefig('figures/synth_energy_distance_vs_d.png', dpi=300)
    plt.close()
    print("Fig 1 generated")
except Exception as e:
    print(f"Error Fig 1: {e}")

# ---------------------------------------------------------
# Figure 2: MI(M) curves for nt=2
# ---------------------------------------------------------
try:
    with open('approach2/mi_curve_results.json', 'r') as f:
        mi_data = json.load(f)
    
    nt_key = 'nt2'
    if nt_key in mi_data:
        data = mi_data[nt_key]
        M_vals = data['M_values']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        methods = {
            'flow': ('Flow', 'Normalizing Flow'),
            'svgd': ('SVGD', 'SVGD'),
            'score': ('Score-based', 'Score-based'),
            'amortized': ('MCMC amorti', 'MCMC Amorti'),
            'pam': ('PAM', 'PAM (Cartésien)'),
            'jeffreys_1d': ('Jeffreys 1D', 'Jeffreys 1D (Baseline)')
        }
        
        for key, (color_key, label) in methods.items():
            if key in data:
                means = []
                m_plot = []
                for m_str in map(str, M_vals):
                    if m_str in data[key]:
                        means.append(data[key][m_str]['mean'])
                        m_plot.append(int(m_str))
                if means:
                    ax.plot(m_plot, means, marker='o', linewidth=2, color=COLORS[color_key], label=label)

        # Plot log2(M) bound
        x_bound = np.linspace(min(M_vals), max(M_vals), 100)
        ax.plot(x_bound, np.log2(x_bound), 'k--', alpha=0.6, label='Borne log2(M)')

        ax.set_xlabel('Taille de la constellation $M$')
        ax.set_ylabel('Information Mutuelle $I(X; Y)$ [bits]')
        ax.set_xscale('log', base=2)
        ax.set_xticks(M_vals)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
        ax.set_title(f'Comparaison de l\'Information Mutuelle vs $M$ (nt=2, d=4)')
        plt.tight_layout()
        plt.savefig('figures/synth_mi_vs_M_nt2.png', dpi=300)
        plt.close()
        print("Fig 2 generated")
    else:
        print(f"Key {nt_key} not in mi_data")
except Exception as e:
    print(f"Error Fig 2: {e}")

# ---------------------------------------------------------
# Figure 3: Complexity Summary
# ---------------------------------------------------------
try:
    with open('approach2/benchmark_logdet_complexity.json', 'r') as f:
        comp_data = json.load(f)

    d_vals = []
    t_vals = []
    
    for key, val in comp_data.items():
        if key.startswith('synth_d') or key.startswith('trained_d'):
            d = int(key.split('_d')[1])
            d_vals.append(d)
            t_vals.append(val['mean'] * 1000) # Convert to ms
            
    # Sort
    d_vals = np.array(d_vals)
    t_vals = np.array(t_vals)
    sorted_idx = np.argsort(d_vals)
    d_vals = d_vals[sorted_idx]
    t_vals = t_vals[sorted_idx]
    
    # Keep only unique d for fit to avoid bias if both synth and trained are present for same d
    # actually let's just plot all points
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot data points
    ax.scatter(d_vals, t_vals, color='black', label='Mesures (log-det jacobien)', zorder=5)
    
    # Fit power law
    try:
        popt, _ = curve_fit(power_law, d_vals, t_vals, p0=[0.1, 3])
        x_fit = np.linspace(min(d_vals), max(d_vals), 100)
        y_fit = power_law(x_fit, *popt)
        ax.plot(x_fit, y_fit, 'r--', label=f'Ajustement: $T \propto d^{{{popt[1]:.2f}}}$')
    except Exception as e:
        print(f"Fit error: {e}")

    ax.set_xlabel('Dimension $d$')
    ax.set_ylabel('Temps d\'exécution par batch [ms]')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    ax.set_title('Complexité Temporelle du Calcul du Log-Déterminant')
    
    # Add clarification L vs d as text
    # Usually L scales with d in trained models (e.g., L ~ d or L ~ d/2)
    textstr = "Note : Le nombre de couches $L$\ncroît généralement avec $d$ dans\nles modèles Flow entraînés."
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('figures/synth_complexity.png', dpi=300)
    plt.close()
    print("Fig 3 generated")
except Exception as e:
    print(f"Error Fig 3: {e}")
