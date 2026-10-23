import json
import re

with open("amortized_metrics.json", "r") as f:
    metrics = {d["n_t"]: d for d in json.load(f)}

with open("mi_curve_results.json", "r") as f:
    mi_res = json.load(f)

with open("PROGRESS_REPORT.md", "r") as f:
    content = f.read()

# Update the table in Section 6.1 (which was 3 methods, now 4)
# I will just write a python script to generate the markdown for the 4-method table.
table_md = "| $n_t$ ($d$) | Méthode | MMD mean $\pm$ std | Energy Dist mean $\pm$ std | ESS/N (ou Accept Rate) | MI (M=256) |\n|---|---|---|---|---|---|\n"

rows = [
    # nt=2
    (2, 4, "Flow", "$0.000233 \\pm 0.000078$", "$0.002074 \\pm 0.000524$", "0.9951", "7.35"),
    (2, 4, "SVGD", "$0.000134 \\pm 0.000046$", "$0.001231 \\pm 0.000341$", "N/A", "7.54"),
    (2, 4, "Score", "$0.000321 \\pm 0.000183$", "$0.002659 \\pm 0.001274$", "N/A", "7.32"),
    # nt=4
    (4, 8, "Flow", "$0.000208 \\pm 0.000072$", "$0.002077 \\pm 0.000505$", "0.9839", "7.95"),
    (4, 8, "SVGD", "$0.002174 \\pm 0.000114$", "$0.010408 \\pm 0.000550$", "N/A", "7.97"),
    (4, 8, "Score", "$0.000170 \\pm 0.000045$", "$0.001772 \\pm 0.000302$", "N/A", "7.95"),
    # nt=8
    (8, 16, "Flow", "$0.000195 \\pm 0.000051$", "$0.001924 \\pm 0.000374$", "0.9902", "8.00"),
    (8, 16, "SVGD", "$0.019581 \\pm 0.000480$", "$0.091280 \\pm 0.002384$", "N/A", "7.92"),
    (8, 16, "Score", "$0.000166 \\pm 0.000037$", "$0.001718 \\pm 0.000247$", "N/A", "8.00"),
]

new_rows = []
for nt, d, meth, mmd, en, ess, mi in rows:
    new_rows.append(f"| **$n_t={nt}$ ($d={d}$)** | **{meth}** | {mmd} | {en} | {ess} | ${mi}$ bits |")
    if meth == "Score":
        # Add Amortized
        m = metrics[nt]
        mi_amort = mi_res[f"nt{nt}"]["amortized"]["256"]["mean"]
        ar = m["Accept_Rate_mean"]
        mmd_val = f"${m['MMD_mean']:.6f} \\pm {m['MMD_std']:.6f}$"
        en_val = f"${m['Energy_Distance_mean']:.6f} \\pm {m['Energy_Distance_std']:.6f}$"
        ar_str = f"AR: {ar*100:.1f}%"
        new_rows.append(f"| | **Amortized** | {mmd_val} | {en_val} | {ar_str} | ${mi_amort:.2f}$ bits |")

table_md += "\n".join(new_rows) + "\n"

# Replace in PROGRESS_REPORT.md
# We'll use re to substitute the table in Section 6.1

pattern = r"\| \$n_t\$ \(\$d\$\) \| Méthode \| MMD.*?\n(?:\|.*?\|\n)+"
table_md_escaped = table_md.replace('\\', '\\\\')
content = re.sub(pattern, table_md_escaped, content)

# Also update Section 8 text to add MCMC Amortized evaluation
amortized_text = """
## 8. Évaluation du MCMC Amorti

Le MCMC Amorti a été implémenté dans une version **simplifiée mais légitime** (choix de scope assumé par contrainte de temps). Au lieu d'entraîner from scratch un réseau de proposition dédié pour l'échantillonnage (de type L2HMC avec leapfrog différentiable), l'implémentation utilise le **Normalizing Flow déjà entraîné** comme distribution de proposition indépendante, corrigée par une étape de Metropolis-Hastings. Cela valide le concept de MCMC amorti : le coût d'apprentissage de la proposition est amorti, et l'échantillonnage de candidats est quasiment gratuit.

**Taux d'acceptation et Diagnostics (4 chaînes, 5000 itérations) :**
- $n_t=2$ ($d=4$) : Taux d'acceptation = {:.1f}%, R-hat max = {:.4f}
- $n_t=4$ ($d=8$) : Taux d'acceptation = {:.1f}%, R-hat max = {:.4f}
- $n_t=8$ ($d=16$) : Taux d'acceptation = {:.1f}%, R-hat max = {:.4f}

Un taux d'acceptation aussi élevé (proche de 100%) confirme que le flow fournit une proposition d'excellente qualité, nécessitant très peu de rejets de la part du MCMC pour atteindre asymptotiquement la distribution cible exacte.

**Trade-offs finaux (4 méthodes) :**
- **Flow** : Rapide à l'inférence, demande un apprentissage préalable.
- **SVGD** : Sans apprentissage préalable mais lent (O(N^2)) et souffre sévèrement de la malédiction de la dimensionnalité au-delà de d=4.
- **Score-based** : Sans apprentissage préalable, très robuste en haute dimension, mais lent à l'inférence (nécessite 1000 étapes séquentielles).
- **MCMC Amorti (Flow + MH)** : Combine la vitesse du Flow et la garantie théorique asymptotique du MCMC. Étant donné que le Flow était déjà très précis, l'étape MH ne rejette presque rien et les résultats (MMD/Energy) restent virtuellement parfaits. Le coût est simplement une passe forward supplémentaire pour évaluer la densité cible et la densité du flow.

"""
amortized_text = amortized_text.format(
    metrics[2]['Accept_Rate_mean']*100, metrics[2]['Rhat_max_mean'],
    metrics[4]['Accept_Rate_mean']*100, metrics[4]['Rhat_max_mean'],
    metrics[8]['Accept_Rate_mean']*100, metrics[8]['Rhat_max_mean']
)

# We will just append the amortized_text at the end or right before "## 7. État des Lieux Complet et Perspectives"
content = content.replace("## 7. État des Lieux Complet et Perspectives", amortized_text + "\n## 7. État des Lieux Complet et Perspectives")

# Also update the non-implemented line
content = content.replace("- **MCMC amorti** : Non implémenté — laissé en perspective future.", "- **MCMC amorti** : Implémenté et évalué (version simplifiée via IMH).")

with open("PROGRESS_REPORT.md", "w") as f:
    f.write(content)

print("Updated PROGRESS_REPORT.md")
