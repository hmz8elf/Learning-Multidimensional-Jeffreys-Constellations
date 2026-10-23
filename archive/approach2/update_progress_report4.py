import re

with open("PROGRESS_REPORT.md", "r") as f:
    content = f.read()

old_text = """  **Analyse de Puissance Statistique (Power Analysis) :**
  Afin de s'assurer que la haute p-value à $n_t=8$ ne provient pas d'une perte de puissance du test de permutation en haute dimension, des perturbations contrôlées ont été appliquées aux échantillons du flow (décalage de 1%, 2%, 5%, 10% de l'écart-type) :
  - À **$n_t=8$**, le test détecte massivement une infime perturbation de **1%** ($p = 0.002$). Le test conserve donc une sensibilité extrême en haute dimension.
  - À **$n_t=4$**, le flow original présentait un micro-biais en direction opposée, qu'une perturbation de 1% à 2% vient compenser (augmentant la p-value), mais une perturbation de **5%** est fermement détectée ($p = 0.0$).
  - À **$n_t=2$**, le test est régulier et détecte fiablement une perturbation dès **2%** à **5%**."""

new_text = """  **Analyse de Puissance Statistique (Power Analysis) :**
  Afin de s'assurer que la haute p-value à $n_t=8$ ne provient pas d'une perte de puissance du test de permutation en haute dimension, des perturbations contrôlées ont été appliquées aux échantillons du flow (décalage de 1%, 2%, 5%, 10% de l'écart-type) :
  - À **$n_t=8$**, le test détecte massivement une infime perturbation de **1%** ($p = 0.002$). Le test conserve donc une sensibilité extrême en haute dimension.
  - À **$n_t=4$**, une anomalie avait été initialement repérée où une faible perturbation (1% à 2%) faisait paradoxalement remonter la p-value (suggérant la compensation d'un "micro-biais").
  - À **$n_t=2$**, le test est régulier et détecte fiablement une perturbation dès **2%** à **5%**.

  **Investigation de l'anomalie de puissance à $n_t=4$ :**
  Pour vérifier si l'effet observé à $n_t=4$ était systématique, l'expérience a été répétée spécifiquement avec les mêmes amplitudes (1% et 2%) sur 10 tirages aléatoires indépendants (impliquant un rééchantillonnage du flow, de la référence MCMC, et un nouveau bruit de perturbation).
  - *Résultats mesurés :* Les 10 p-values obtenues pour la perturbation de 1% varient massivement dans les deux sens sans tendance claire (ex: 0.618, 0.013, 0.833, 0.053, 0.456, ...). Il en va de même pour la perturbation de 2%.
  - *Interprétation :* La très forte variance des p-values indique que l'augmentation isolée observée initialement n'est pas un biais directionnel reproductible du modèle, mais relève uniquement du bruit d'échantillonnage ponctuel (non reproductible). En dimension $d=8$, avec $N=5000$, la distance entre la distribution du flow et celle de la cible est de l'ordre de l'erreur d'échantillonnage, rendant les p-values extrêmement volatiles au moindre tirage. Le flow à $n_t=4$ ne présente donc pas de défaut compensable, et l'anomalie initiale n'est qu'un artefact statistique propre à la première graine aléatoire testée."""

content = content.replace(old_text, new_text)

with open("PROGRESS_REPORT.md", "w") as f:
    f.write(content)

