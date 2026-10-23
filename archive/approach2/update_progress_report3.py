import re

with open("PROGRESS_REPORT.md", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if line.startswith("- **Profiling de complexité computationnelle (Log-Déterminant vs Dimension)**"):
        skip = True
        new_lines.append("""- **Profiling de complexité computationnelle (Log-Déterminant vs Dimension & Architecture)** : 
  Le temps de calcul du log-déterminant du jacobien lors du passage forward a été mesuré sur GPU (batch de 1000). 
  
  **1. Résultats initiaux (effet combiné dimension $d$ et architecture $L, h$) :**
  - Modèles entraînés ($n_t=2, 4, 8$) où $d$ augmente (4, 8, 16) conjointement avec l'architecture ($L=8, 10, 12$ et $h=64, 128, 128$) : temps croissants de $2.079$ ms à $3.005$ ms (exposant apparent $b \\approx 0.27$).
  - Modèles synthétiques (architecture fixe $L=8$, $h=64$) sur la même plage de $d$ (2 à 64) : temps constants autour de $\\sim 1.95$ ms (exposant empirique $b \\approx 0.00$). 
  À partir de $d=1024$, le temps commence à croître linéairement (ex: 14.12 ms pour $d=4096$, 62.16 ms pour $d=16384$).

  **2. Expérience de contrôle (isolement de l'effet de l'architecture à $d=8$ fixe) :**
  Pour vérifier l'origine de l'anomalie d'augmentation observée sur les modèles entraînés, des modèles synthétiques ont été testés en fixant $d=8$ constant mais en reproduisant séquentiellement les architectures des modèles entraînés :
  - Architecture $n_t=2$ ($L=8, h=64$) à $d=8$ : $1.986 \\pm 0.043$ ms
  - Architecture $n_t=4$ ($L=10, h=128$) à $d=8$ : $2.454 \\pm 0.046$ ms
  - Architecture $n_t=8$ ($L=12, h=128$) à $d=8$ : $2.960 \\pm 0.064$ ms

  **INTERPRÉTATION MISE À JOUR :**
  Les mesures de contrôle isolées (à $d$ fixe) révèlent que l'augmentation de temps générée par le seul alourdissement de l'architecture ($1.986 \\to 2.454 \\to 2.960$ ms) correspond presque parfaitement à l'augmentation globale initialement observée sur les modèles entraînés ($2.079 \\to 2.531 \\to 3.005$ ms). 
  Ces résultats confirment que l'effet observé n'était pas une anomalie de dimensionnalité, mais le reflet direct de l'augmentation du nombre de couches de couplage $L$ (et de la largeur $h$). La variation de la dimension $d$ dans cette plage a un effet mesurable négligeable par rapport au coût d'itération des couches.
  
  **Conclusion finale sur la scalabilité** : 
  Pour des dimensions modérées ($d \\le 64$), les résultats valident empiriquement une complexité en temps d'exécution de $\\mathcal{O}(L)$ (linéaire en profondeur) et de $\\mathcal{O}(d^0)$ (indépendante de la dimension d'entrée), le GPU masquant les calculs tensoriels sous un overhead fixe dominé par les lancements séquentiels de noyaux CUDA. 
  Pour des dimensions massives ($d \\ge 1024$), la saturation en calcul du GPU lève ce masquage, et les temps mesurés rejoignent le régime asymptotique théorique linéaire en $\\mathcal{O}(d^1)$ attendu pour une couche dense.
""")
        continue
    if skip and line.startswith("**Tableau comparatif MIMO $n_t=2$ ($d=4$)"):
        skip = False
    
    if not skip:
        if line.startswith("- [x] **Profiling de complexité computationnelle**"):
            new_lines.append("- [x] **Profiling de complexité computationnelle** : Livrable terminé. L'étude combinée sur la dimension $d$ et la profondeur $L$ a profilé empiriquement le `logdet` et clarifié les régimes de complexité asymptotique et d'overhead GPU (voir Section 3).\n")
        else:
            new_lines.append(line)

with open("PROGRESS_REPORT.md", "w") as f:
    f.writelines(new_lines)

