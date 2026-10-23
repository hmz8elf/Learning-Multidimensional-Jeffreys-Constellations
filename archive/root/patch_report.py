import re
with open("Approche 2/mi_curve.py", "r") as f:
    content = f.read()

old_report = """    lines = [
        "",
        "- [x] **Courbe d'Information Mutuelle (MI) vs $M$** : Livrable terminé.",
        f"  Courbe MI vs M calculée pour $n_t=2,4,8$ avec Blahut-Arimoto",
        f"  (max\\_iter={BA_MAX_ITER}, tol={BA_TOL}), {N_SEEDS} graines, canal AWGN σ²={SIGMA2}.",
        "  Baselines : PAM Cartésienne (toutes dimensions) + Jeffreys 1D Cartésienne (nt=2 seulement).",
        "  Figures : `mi_curve_all.png`, `mi_curve_per_nt.png`. Résultats JSON : `mi_curve_results.json`.",
        "",
    ]"""

new_report = """    lines = [
        "",
        "- [x] **Courbe d'Information Mutuelle (MI) vs $M$** : Livrable terminé.",
        f"  Courbe MI vs M calculée pour $n_t=2,4,8$ avec Blahut-Arimoto",
        f"  (max\\_iter={BA_MAX_ITER}, tol={BA_TOL}), {N_SEEDS} graines, canal AWGN σ²={SIGMA2}.",
        "  Baselines : PAM Cartésienne (toutes dimensions) + Jeffreys 1D Cartésienne (nt=2 seulement).",
        "  **LIMITATION PAM HAUTE DIMENSION** : Pour $d \\ge 16$ ($n_t \\ge 8$), $M_{1d}$ reste fixé à 2 pour $M \\le 256$. "
        "  La PAM est donc un sous-échantillon aléatoire de $M$ points d'une grille QPSK multidimensionnelle. "
        "  C'est une limitation acceptée pour ce livrable.",
        "  Figures : `mi_curve_all.png`, `mi_curve_per_nt.png`. Résultats JSON : `mi_curve_results.json`.",
        "",
    ]"""

content = content.replace(old_report, new_report)

with open("Approche 2/mi_curve.py", "w") as f:
    f.write(content)
print("Patched report section")
