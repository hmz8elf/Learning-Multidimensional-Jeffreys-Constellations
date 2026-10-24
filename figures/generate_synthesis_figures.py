"""Synthesis figures, generated only from the committed JSON result files.

Run from the repository root:
    python figures/generate_synthesis_figures.py

Outputs:
    figures/synth_energy_distance_vs_d.png   energy distance vs dimension
    figures/synth_mi_vs_M_nt{2,4,8}.png      mutual information vs M, one figure per n_t
    figures/synth_complexity.png             flow log-det time vs dimension
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Harmonized styling. Each method also gets its own marker and line style,
# so the curves stay distinguishable without color (colorblind / print).
STYLES = {
    'Flow':           dict(color='#1f77b4', marker='o', linestyle='-'),
    'SVGD':           dict(color='#ff7f0e', marker='s', linestyle='--'),
    'Score-based':    dict(color='#2ca02c', marker='^', linestyle='-.'),
    'Amortized MCMC': dict(color='#d62728', marker='D', linestyle=':'),
    'PAM':            dict(color='#7f7f7f', marker='x', linestyle='-'),
}
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'legend.fontsize': 11,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11})

os.makedirs('figures', exist_ok=True)


def load(path):
    with open(path, 'r') as f:
        return json.load(f)


# ---------------------------------------------------------
# Figure 1: Energy distance vs dimension d
# ---------------------------------------------------------
try:
    sources = {
        'Flow': 'approach2/evaluation_metrics.json',
        'SVGD': 'approach2/svgd_metrics.json',
        'Score-based': 'approach2/score_metrics.json',
        'Amortized MCMC': 'approach2/amortized_metrics.json',
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, path in sources.items():
        rows = [r for r in load(path) if 'd' in r and 'Energy_Distance_mean' in r]
        rows.sort(key=lambda r: r['d'])
        d_vals = [r['d'] for r in rows]
        ed_vals = [r['Energy_Distance_mean'] for r in rows]
        ax.plot(d_vals, ed_vals, linewidth=2, markersize=8, label=name, **STYLES[name])

    ax.set_xlabel('Dimension $d = 2 n_t$')
    ax.set_ylabel('Energy distance to HMC reference (mean of 5 seeds)')
    ax.set_yscale('log')
    ax.set_xticks([4, 8, 16])
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    ax.set_title('Sample quality vs dimension (lower is better)')
    plt.tight_layout()
    plt.savefig('figures/synth_energy_distance_vs_d.png', dpi=300)
    plt.close()
    print("Fig 1 generated")
except Exception as e:
    print(f"Error Fig 1: {e}")

# ---------------------------------------------------------
# Figure 2: MI(M) curves, one figure per n_t
# The "jeffreys_1d" entry of the JSON is not plotted: it uses M_1d^d points
# instead of M, so it is not comparable (README, limitation B3).
# ---------------------------------------------------------
try:
    mi_data = load('approach2/mi_curve_results.json')

    methods = {
        'flow': 'Flow',
        'svgd': 'SVGD',
        'score': 'Score-based',
        'amortized': 'Amortized MCMC',
        'pam': 'PAM',
    }

    for nt in (2, 4, 8):
        nt_key = f'nt{nt}'
        if nt_key not in mi_data:
            print(f"Key {nt_key} not in mi_data")
            continue
        data = mi_data[nt_key]
        M_vals = data['M_values']

        fig, ax = plt.subplots(figsize=(8, 6))

        for key, name in methods.items():
            if key not in data:
                continue
            m_plot = [M for M in M_vals if str(M) in data[key]]
            means = [data[key][str(M)]['mean'] for M in m_plot]
            if means:
                label = 'PAM (Cartesian grid)' if key == 'pam' else name
                ax.plot(m_plot, means, linewidth=2, markersize=7, label=label, **STYLES[name])

        ax.plot(M_vals, np.log2(M_vals), 'k--', alpha=0.6, linewidth=1.2,
                label='Upper bound $\\log_2 M$')

        ax.set_xlabel('Constellation size $M$')
        ax.set_ylabel('Mutual information $I(X;Y)$ [bits]')
        ax.set_xscale('log', base=2)
        ax.set_xticks(M_vals)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(loc='upper left')
        ax.set_title(f'Mutual information vs $M$ ($n_t = {nt}$, $d = {2 * nt}$)')
        plt.tight_layout()
        plt.savefig(f'figures/synth_mi_vs_M_nt{nt}.png', dpi=300)
        plt.close()
        print(f"Fig 2 (nt={nt}) generated")
except Exception as e:
    print(f"Error Fig 2: {e}")

# ---------------------------------------------------------
# Figure 3: Flow log-det complexity
# Fixed architecture (L=8, h=64) and trained models (L grows with d) are
# plotted as separate series; the fit is the "fit_synth" entry of the JSON.
# ---------------------------------------------------------
try:
    comp_data = load('approach2/benchmark_logdet_complexity.json')

    def series(prefix):
        pts = sorted((int(k.split('_d')[1]), v['mean'] * 1000, v['L'])
                     for k, v in comp_data.items() if k.startswith(prefix))
        return zip(*pts)

    fig, ax = plt.subplots(figsize=(8, 6))

    d_s, t_s, _ = series('synth_d')
    ax.plot(d_s, t_s, color='black', marker='o', linestyle='-', linewidth=1.5,
            label='Fixed architecture (L = 8, h = 64)', zorder=5)

    d_t, t_t, L_t = series('trained_d')
    ax.plot(d_t, t_t, color='#1f77b4', marker='s', linestyle='--', linewidth=1.5,
            markersize=8, label='Trained flows (L = 8 / 10 / 12)', zorder=6)
    for d, t, L in zip(d_t, t_t, L_t):
        ax.annotate(f'L={L}', (d, t), textcoords='offset points', xytext=(6, 4), fontsize=10)

    fit = comp_data['fit_synth']
    b = round(fit['b'], 2) + 0.0  # avoid printing "-0.00"
    x_fit = np.array([min(d_s), max(d_s)], dtype=float)
    ax.plot(x_fit, np.exp(fit['log_a']) * x_fit ** fit['b'] * 1000, 'r:', linewidth=1.5,
            label=f'Power-law fit, fixed arch.: $T \\propto d^{{{b:.2f}}}$')

    ax.set_xlabel('Dimension $d$')
    ax.set_ylabel('Time per batch of 1000 [ms]')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks([2, 4, 8, 16, 32, 64])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks([2.0, 2.5, 3.0])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(plt.NullFormatter())
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc='upper left')
    ax.set_title('Flow log-determinant time vs dimension (GPU)')
    plt.tight_layout()
    plt.savefig('figures/synth_complexity.png', dpi=300)
    plt.close()
    print("Fig 3 generated")
except Exception as e:
    print(f"Error Fig 3: {e}")
