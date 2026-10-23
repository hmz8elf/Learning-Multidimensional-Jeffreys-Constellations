import re

with open("PROGRESS_REPORT.md", "r") as f:
    content = f.read()

old_section = """- **Profiling de complexité computationnelle (Log-Déterminant vs Dimension)** : 
  Le temps de calcul du log-déterminant du jacobien lors du passage forward a été mesuré sur GPU (batch de 1000) en fonction de la dimension $d$. 
  
  **Résultats bruts (modèles entraînés) :**
  - $n_t=2$ ($d=4$, 8 couches) : $2.079 \\pm 0.370$ ms
  - $n_t=4$ ($d=8$, 10 couches) : $2.531 \\pm 0.270$ ms
  - $n_t=8$ ($d=16$, 12 couches) : $3.005 \\pm 0.247$ ms
  
  **Résultats bruts (modèles synthétiques à architecture fixe, 8 couches) :**
  Les temps mesurés restent constants autour de $\\sim 1.95$ ms pour toutes les dimensions de $d=2$ à $d=64$. À partir de $d=1024$, le temps commence à croître linéairement (ex: 2.98 ms pour $d=1024$, 14.12 ms pour $d=4096$, 62.16 ms pour $d=16384$).

  **Ajustement de la loi de puissance (temps $\\approx a \\cdot d^b$) :**
  - Sur les modèles synthétiques à architecture fixe (pour $d \\le 64$) : exposant empirique $b \\approx 0.00$.
  - Sur les modèles entraînés (architectures variables) : exposant empirique $b \\approx 0.27$.

  **INTERPRÉTATION :**
  Les résultats sont compatibles avec l'hypothèse selon laquelle, pour les faibles dimensions ($d \\le 64$), le temps de calcul sur GPU est dominé par l'overhead fixe lié aux lancements de noyaux CUDA (kernel launch latency), ce qui masque la complexité théorique en $\\mathcal{O}(d)$ de l'architecture RealNVP. Cela suggère que la complexité empirique par rapport à la dimension est de $\\mathcal{O}(d^0)$ (constante) dans ce régime. Pour des dimensions beaucoup plus grandes ($d \\ge 1024$), les temps mesurés suggèrent l'émergence d'une complexité linéaire en $\\mathcal{O}(d^1)$, ce qui est cohérent avec la complexité théorique attendue pour une couche dense de type MLP.
  
  *Anomalie identifiée dans l'évaluation des modèles entraînés :* La croissance du temps de calcul observée pour les modèles entraînés ($b \\approx 0.27$) constitue une anomalie par rapport à l'exposant quasi-nul des modèles synthétiques pour les mêmes dimensions. Ce point nécessite une vérification supplémentaire. Nous proposons deux explications possibles pour cette non-constance :
  1. (Liée à l'architecture) : Les modèles entraînés utilisent un nombre croissant de couches de couplage ($L=8$ pour $n_t=2$, $L=10$ pour $n_t=4$, $L=12$ pour $n_t=8$). L'augmentation du temps de calcul observée ($\\sim 2.08$ ms à $\\sim 3.01$ ms) serait causée par l'ajout de ces couches supplémentaires (qui requièrent des lancements CUDA séquentiels), et non par l'augmentation de la dimension $d$ elle-même.
  2. (Liée à la méthode de mesure) : Une variabilité résiduelle liée à la gestion de la mémoire GPU ou des fluctuations d'horloge (throttling) lors du chargement des différents checkpoints pourrait affecter sélectivement certains modèles entraînés, générant un biais apparent de dimensionnalité."""

new_section = """- **Profiling de complexité computationnelle (Log-Déterminant vs Dimension & Architecture)** : 
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
  Les mesures de contrôle isolées (à $d$ fixe) révèlent que l'augmentation de temps générée par le seul alourdissement de l'architecture ($1.986 \\to 2.454 \\to 2.960$ ms) explique la quasi-totalité de l'augmentation globale initialement observée sur les modèles entraînés ($2.079 \\to 2.531 \\to 3.005$ ms). 
  Ces résultats confirment que l'effet observé n'était pas une anomalie de dimensionnalité, mais le reflet direct du nombre de couches de couplage $L$. La variation de la dimension $d$ dans cette plage a un effet mesurable négligeable par rapport au coût d'itération des couches.
  
  **Conclusion finale sur la scalabilité** : 
  Pour des dimensions modérées ($d \\le 64$), les résultats valident empiriquement une complexité en temps d'exécution de $\\mathcal{O}(L)$ (linéaire en profondeur) et de $\\mathcal{O}(d^0)$ (indépendante de la largeur), le GPU masquant les calculs tensoriels sous un overhead fixe dominé par les lancements séquentiels de noyaux CUDA. 
  Pour des dimensions massives ($d \\ge 1024$), la saturation en calcul du GPU lève ce masquage, et les temps mesurés rejoignent le régime asymptotique théorique linéaire en $\\mathcal{O}(d^1)$ attendu pour une couche dense."""

content = content.replace(old_section, new_section)

# Update the bullet point in section 4
old_bullet = "- [x] **Profiling de complexité computationnelle** : Le temps de calcul du `logdet` en fonction de la dimension $d$ a été profilé empiriquement (voir Section 3)."
new_bullet = "- [x] **Profiling de complexité computationnelle** : Livrable terminé. L'étude combinée sur la dimension $d$ et la profondeur $L$ a profilé empiriquement le `logdet` et clarifié les régimes de complexité asymptotique et d'overhead GPU (voir Section 3)."
content = content.replace(old_bullet, new_bullet)

with open("PROGRESS_REPORT.md", "w") as f:
    f.write(content)

