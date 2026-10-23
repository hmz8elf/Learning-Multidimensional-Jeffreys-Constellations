"""
generer_figures_nb.py — Figures du rapport en noir et blanc strict (PDF).

Sorties (noms attendus par les \\includegraphics du chapitre 4) :
  ../Rapport/figures/complexite_baseline.pdf
  ../Rapport/figures/distance_energie.pdf
  ../Rapport/figures/mi_en_fonction_de_M.pdf
  ../Rapport/figures/courbes_perte_flow.pdf

Aucune couleur : l'identité des séries est portée par le marqueur et le style
de ligne. Toutes les valeurs sont lues dans les fichiers de résultats, sauf les
temps de la baseline (voir TEMPS_BASELINE).

Lancer depuis le dossier CNRST/ :  python generer_figures_nb.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ICI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ICI, "Approche 2")
SORTIE = os.path.join(ICI, "..", "Rapport", "figures")
os.makedirs(SORTIE, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.2,
    "lines.markersize": 5.5,
    "axes.edgecolor": "black",
    "axes.prop_cycle": plt.cycler(color=["black"]),
    "pdf.fonttype": 42,
})

# Style par méthode : (libellé, marqueur, style de ligne, remplissage du marqueur)
STYLE = {
    "flow":        ("Flow (RealNVP)",        "o", "-",               "black"),
    "svgd":        ("SVGD",                  "s", "--",              "white"),
    "score":       ("ALD",                   "^", "-.",              "black"),
    "amortized":   ("MCMC amorti",           "D", ":",               "white"),
    "pam":         ("PAM cartésienne",       "x", (0, (5, 1, 1, 1)), "black"),
    "jeffreys_1d": ("Jeffreys 1D cartésienne", "v", (0, (1, 3)),     "white"),
}


def trace(ax, key, x, y, yerr=None):
    lab, mk, ls, fc = STYLE[key]
    ax.errorbar(x, y, yerr=yerr, label=lab, marker=mk, linestyle=ls,
                color="black", markerfacecolor=fc, markeredgecolor="black",
                capsize=2.5 if yerr is not None else 0, elinewidth=0.8)


class VirguleFormatter(ScalarFormatter):
    """Graduations à virgule décimale (la locale fr_FR n'étant pas installée,
    axes.formatter.use_locale ne peut pas servir)."""
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def grille(ax):
    ax.grid(True, which="major", color="0.85", linewidth=0.5)
    ax.set_axisbelow(True)


def charger(nom):
    with open(os.path.join(RES, nom)) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# Figure 1 : complexité de la baseline par règle de chaîne
# ─────────────────────────────────────────────────────────────
# ATTENTION : ces trois temps ne figurent dans AUCUN fichier de résultats du
# dépôt (benchmark_chain_rule_vs_d.py n'enregistre rien). Ils sont repris tels
# quels du rapport (chapitre3_methodologie.tex, tableau baseline-complexite).
# Les temps pour d=2 et d=3 ne sont pas disponibles : ils ne sont pas tracés.
TEMPS_BASELINE = {1: 0.014, 4: 4.166, 5: 33.01}  # secondes, M=4, n_quad=8


def figure_complexite():
    d = np.array(sorted(TEMPS_BASELINE), dtype=float)
    t = np.array([TEMPS_BASELINE[int(k)] for k in d])
    pente, c0 = np.polyfit(d, np.log(t), 1)       # ln t = c0 + pente * d
    b4, b8 = pente / np.log(4), pente / np.log(8)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(d, t, "o", color="black", markerfacecolor="black",
            label="Temps mesurés ($M=4$, $n_{\\mathrm{quad}}=8$)", zorder=3)
    dd = np.linspace(1, 5, 50)
    ax.plot(dd, np.exp(c0 + pente * dd), "--", color="black",
            label=("Ajustement : $b\\approx" + f"{b4:.2f}".replace(".", "{,}") + "$ (base 4), $"
                   + f"{b8:.2f}".replace(".", "{,}") + "$ (base 8)"))
    ax.set_yscale("log")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Temps de construction (s)")
    grille(ax)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(SORTIE, "complexite_baseline.pdf"))
    plt.close(fig)
    return b4, b8


# ─────────────────────────────────────────────────────────────
# Figure 2 : distance d'énergie en fonction de n_t
# ─────────────────────────────────────────────────────────────
DECALAGE = {"flow": 0.94, "svgd": 0.98, "score": 1.02, "amortized": 1.06}


def figure_energie():
    fichiers = {
        "flow": "evaluation_metrics.json",
        "svgd": "svgd_metrics.json",
        "score": "score_metrics.json",
        "amortized": "amortized_metrics.json",
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    valeurs = {}
    for key, nom in fichiers.items():
        lignes = sorted(charger(nom), key=lambda r: r["n_t"])
        nt = [r["n_t"] for r in lignes]
        m = [r["Energy_Distance_mean"] for r in lignes]
        s = [r["Energy_Distance_std"] for r in lignes]
        valeurs[key] = dict(zip(nt, zip(m, s)))
        trace(ax, key, [n * DECALAGE[key] for n in nt], m, s)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([2, 4, 8])
    ax.set_xticklabels(["2\n($d=4$)", "4\n($d=8$)", "8\n($d=16$)"])
    ax.minorticks_off()
    ax.set_xlabel("$n_t$")
    ax.set_ylabel("Distance d'énergie")
    grille(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(SORTIE, "distance_energie.pdf"))
    plt.close(fig)
    return valeurs


# ─────────────────────────────────────────────────────────────
# Figure 3 : MI en fonction du M réel, n_t = 2, 4, 8
# ─────────────────────────────────────────────────────────────
def m_reel(res, key, M):
    """Nombre de points réellement présents dans la constellation évaluée.
    PAM et Jeffreys 1D stockent M_effective ; les quatre méthodes génèrent
    exactement M points (flow.sample(M), SVGD à M particules, ALD à M
    particules, MCMC amorti à M échantillons)."""
    return int(res[key][str(M)].get("M_effective", M))


def figure_mi():
    data = charger("mi_curve_results.json")
    ordre = ["flow", "svgd", "score", "amortized", "pam", "jeffreys_1d"]
    violations, points = [], {}

    # Rangée du haut : MI ; rangée du bas : écart à la borne log2 M - MI
    # (axes séparés, pas de double axe). Sans la rangée du bas, toutes les
    # courbes se confondent avec la borne à n_t = 4 et 8.
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1]})
    for j, nt in enumerate([2, 4, 8]):
        ax, axd = axes[0, j], axes[1, j]
        res = data[f"nt{nt}"]
        M_dem = res["M_values"]
        xb = np.array([4, 256])
        ax.plot(xb, np.log2(xb), color="black", linewidth=0.6,
                linestyle=(0, (2, 2)), label="Borne $\\log_2 M$")
        axd.axhline(0, color="black", linewidth=0.6, linestyle=(0, (2, 2)))
        for key in ordre:
            if key not in res:
                continue
            xy = {}
            for M in M_dem:
                e = res[key].get(str(M))
                if e is None or e.get("mean") is None:
                    continue
                Mr = m_reel(res, key, M)
                mi, sd = e["mean"], e.get("std", 0.0)
                points[(nt, key, M)] = (Mr, mi, sd)
                if mi > np.log2(Mr) + 1e-9:
                    violations.append((nt, key, M, Mr, mi, np.log2(Mr)))
                # Plusieurs M demandés peuvent donner le même M réel (PAM, Jeffreys 1D) :
                # on ne garde qu'un point par M réel (valeurs identiques à 1e-3 près).
                xy.setdefault(Mr, (mi, sd))
            x = sorted(xy)
            mi = np.array([xy[k][0] for k in x])
            sd = np.array([xy[k][1] for k in x])
            trace(ax, key, x, mi, sd)
            trace(axd, key, x, np.log2(x) - mi, sd)
        for a in (ax, axd):
            a.set_xscale("log", base=2)
            a.set_xticks([4, 8, 16, 32, 64, 128, 256])
            a.set_xticklabels(["4", "8", "16", "32", "64", "128", "256"])
            a.minorticks_off()
            a.yaxis.set_major_formatter(VirguleFormatter())
            grille(a)
        ax.set_ylim(1.5, 8.4)
        axd.set_ylim(-0.03, 0.72)
        ax.set_title(f"$n_t={nt}$ ($d={2 * nt}$)", fontsize=10)
        axd.set_xlabel("Nombre réel de points $M$")
    axes[0, 0].set_ylabel("Information mutuelle (bits)")
    axes[1, 0].set_ylabel("$\\log_2 M - \\mathrm{MI}$ (bits)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(SORTIE, "mi_en_fonction_de_M.pdf"))
    plt.close(fig)
    return points, violations


# ─────────────────────────────────────────────────────────────
# Figure 4 : courbes de perte (KL inverse à une constante près) du Flow
# ─────────────────────────────────────────────────────────────
def figure_pertes():
    """Une valeur de perte toutes les 100 itérations (train.py). Un panneau par
    n_t, chacun avec son axe y : les niveaux diffèrent d'une dimension à l'autre
    (constante log Z inconnue), seule l'évolution au sein d'un panneau compte."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    stats = {}
    for ax, nt in zip(axes, [2, 4, 8]):
        with open(os.path.join(RES, "checkpoints", f"history_mimo_nt{nt}.json")) as f:
            loss = np.array(json.load(f)["loss"])
        it = 100 * np.arange(1, len(loss) + 1)
        ax.plot(it, loss, color="black", linewidth=1.0, marker="o", markersize=2.5)
        ax.set_title(f"$n_t={nt}$ ($d={2 * nt}$)", fontsize=10)
        ax.set_xlabel("Itération")
        ax.yaxis.set_major_formatter(VirguleFormatter())
        grille(ax)
        stats[nt] = (len(loss) * 100, loss[0], loss[-1], loss.min(), loss.max())
    axes[0].set_ylabel("Perte (à une constante près)")
    fig.tight_layout()
    fig.savefig(os.path.join(SORTIE, "courbes_perte_flow.pdf"))
    plt.close(fig)
    return stats


if __name__ == "__main__":
    b4, b8 = figure_complexite()
    print(f"[complexite] b(base 4)={b4:.3f}  b(base 8)={b8:.3f}  (3 points : d=1,4,5)")

    ed = figure_energie()
    for k, v in ed.items():
        print(f"[energie] {k:10s} " + "  ".join(f"nt={n}: {m:.6f}±{s:.6f}" for n, (m, s) in v.items()))

    for nt, (n, a, b, lo, hi) in figure_pertes().items():
        print(f"[pertes] nt={nt}: {n} it, première={a:.4f} dernière={b:.4f} min={lo:.4f} max={hi:.4f}")

    pts, viol = figure_mi()
    print("[MI] points où M réel ≠ M demandé :")
    for (nt, key, M), (Mr, mi, sd) in sorted(pts.items()):
        if Mr != M:
            print(f"   nt={nt} {key:11s} M demandé={M:3d} → M réel={Mr:3d}  MI={mi:.4f}  log2(M réel)={np.log2(Mr):.4f}")
    if viol:
        print("[MI] VIOLATIONS de MI ≤ log2(M réel) :")
        for v in viol:
            print("   nt=%d %s M=%d Mréel=%d MI=%.6f > %.6f" % v)
        sys.exit(1)
    n = len(pts)
    marge = min(np.log2(Mr) - mi for (Mr, mi, _) in pts.values())
    print(f"[MI] Vérification OK : MI ≤ log2(M réel) pour les {n} points (marge min = {marge:.2e} bit)")
    # Contrôle complémentaire : avec le M demandé (et non réel), la borne serait violée
    faux = [(k, v) for k, v in pts.items() if v[1] > np.log2(k[2]) + 1e-9]
    for (nt, key, M), (Mr, mi, _) in faux:
        print(f"   (si l'on tracait en M demandé : nt={nt} {key} M={M} MI={mi:.4f} > log2 M={np.log2(M):.4f})")
    print("Figures écrites dans", os.path.abspath(SORTIE))
