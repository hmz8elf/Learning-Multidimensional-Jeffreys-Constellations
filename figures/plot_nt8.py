import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Harmonized styling
COLORS = {
    'Flow': '#1f77b4',       # blue
    'SVGD': '#ff7f0e',       # orange
    'Score-based': '#2ca02c', # green
    'MCMC amorti': '#d62728', # red
    'PAM': '#7f7f7f',        # grey
}
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'legend.fontsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11})

os.makedirs('figures', exist_ok=True)

try:
    with open('approach2/mi_curve_results.json', 'r') as f:
        mi_data = json.load(f)
    
    nt_key = 'nt8'
    if nt_key in mi_data:
        data = mi_data[nt_key]
        M_vals = data['M_values']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        methods = {
            'flow': ('Flow', 'Normalizing Flow'),
            'svgd': ('SVGD', 'SVGD (Dégradation en haute dim)'),
            'score': ('Score-based', 'Score-based'),
            'amortized': ('MCMC amorti', 'MCMC Amorti'),
            'pam': ('PAM', 'PAM (Cartésien)')
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
        ax.plot(x_bound, np.log2(x_bound), 'k--', alpha=0.6, label='Borne théorique log2(M)')

        ax.set_xlabel('Taille de la constellation $M$')
        ax.set_ylabel('Information Mutuelle $I(X; Y)$ [bits]')
        ax.set_xscale('log', base=2)
        ax.set_xticks(M_vals)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
        ax.set_title(f'Dégradation de SVGD en haute dimension : I(X;Y) vs M (nt=8, d=16)')
        plt.tight_layout()
        plt.savefig('figures/synth_mi_vs_M_nt8.png', dpi=300)
        plt.close()
        print("Fig 2bis (nt=8) generated")
    else:
        print(f"Key {nt_key} not in mi_data")
except Exception as e:
    print(f"Error: {e}")
