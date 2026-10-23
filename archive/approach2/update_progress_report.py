import re

with open("PROGRESS_REPORT.md", "r") as f:
    content = f.read()

# 1. Update the "Ce qui N'EST PAS encore fait" section
# Replace the unchecked box for profiling with a checked box, or just remove it and put it in section 3.
old_profiling = "- **Profiling de complexité computationnelle** : L'étude de scalabilité requise n'a pas été menée. Le temps par itération et le profilage de coût du calcul du `logdet` (`slogdet`) face au reste du pas forward en fonction de la dimension $d$ n'existent pas."

# We will just remove it from section 4 since we will add it to section 3.
content = content.replace(old_profiling, "- [x] **Profiling de complexité computationnelle** : Le temps de calcul du `logdet` en fonction de la dimension $d$ a été profilé empiriquement (voir Section 3).")

# 2. Add the results to Section 3
results_text = """
- **Profiling de complexité computationnelle (Log-Déterminant vs Dimension)** : 
  Le temps de calcul du log-déterminant du jacobien lors du passage forward a été mesuré sur GPU (batch de 1000) en fonction de la dimension $d$. 
  
  **Résultats bruts (modèles entraînés) :**
  - $n_t=2$ ($d=4$, 8 couches) : $2.079 \pm 0.370$ ms
  - $n_t=4$ ($d=8$, 10 couches) : $2.531 \pm 0.270$ ms
  - $n_t=8$ ($d=16$, 12 couches) : $3.005 \pm 0.247$ ms
  
  **Résultats bruts (modèles synthétiques à architecture fixe, 8 couches) :**
  Les temps mesurés restent constants autour de $\sim 1.95$ ms pour toutes les dimensions de $d=2$ à $d=64$. À partir de $d=1024$, le temps commence à croître linéairement (ex: 2.98 ms pour $d=1024$, 14.12 ms pour $d=4096$, 62.16 ms pour $d=16384$).

  **Ajustement de la loi de puissance (temps $\approx a \cdot d^b$) :**
  - Sur les modèles synthétiques à architecture fixe (pour $d \le 64$) : exposant empirique $b \approx 0.00$.
  - Sur les modèles entraînés (architectures variables) : exposant empirique $b \approx 0.27$.

  **INTERPRÉTATION :**
  Les résultats sont compatibles avec l'hypothèse selon laquelle, pour les faibles dimensions ($d \le 64$), le temps de calcul sur GPU est dominé par l'overhead fixe lié aux lancements de noyaux CUDA (kernel launch latency), ce qui masque la complexité théorique en $\mathcal{O}(d)$ de l'architecture RealNVP. Cela suggère que la complexité empirique par rapport à la dimension est de $\mathcal{O}(d^0)$ (constante) dans ce régime. Pour des dimensions beaucoup plus grandes ($d \ge 1024$), les temps mesurés suggèrent l'émergence d'une complexité linéaire en $\mathcal{O}(d^1)$, ce qui est cohérent avec la complexité théorique attendue pour une couche dense de type MLP.
  
  *Anomalie identifiée dans l'évaluation des modèles entraînés :* La croissance du temps de calcul observée pour les modèles entraînés ($b \approx 0.27$) constitue une anomalie par rapport à l'exposant quasi-nul des modèles synthétiques pour les mêmes dimensions. Ce point nécessite une vérification supplémentaire. Nous proposons deux explications possibles pour cette non-constance :
  1. (Liée à l'architecture) : Les modèles entraînés utilisent un nombre croissant de couches de couplage ($L=8$ pour $n_t=2$, $L=10$ pour $n_t=4$, $L=12$ pour $n_t=8$). L'augmentation du temps de calcul observée ($\sim 2.08$ ms à $\sim 3.01$ ms) serait causée par l'ajout de ces couches supplémentaires (qui requièrent des lancements CUDA séquentiels), et non par l'augmentation de la dimension $d$ elle-même.
  2. (Liée à la méthode de mesure) : Une variabilité résiduelle liée à la gestion de la mémoire GPU ou des fluctuations d'horloge (throttling) lors du chargement des différents checkpoints pourrait affecter sélectivement certains modèles entraînés, générant un biais apparent de dimensionnalité.
"""

# Insert the results text before the comparison table
insertion_point = "**Tableau comparatif MIMO $n_t=2$"
content = content.replace(insertion_point, results_text + "\n" + insertion_point)

with open("PROGRESS_REPORT.md", "w") as f:
    f.write(content)

