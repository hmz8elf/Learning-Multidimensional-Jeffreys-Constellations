# Debug Log - Semaine 3

## ETAPE 8 : Cas 1D
- **Validation 1D** : PASS
- **ESS/N** : 0.9995
- **Commentaires** : Le cas 1D AWGN a un $J(\theta)$ constant et un coût quadratique, ce qui fait que le prior de Jeffreys tilté est une distribution Gaussienne parfaite. La dégénérescence du RealNVP en 1D (transformation affine globale) est un approximateur parfait pour une Gaussienne. L'ESS est donc excellent (~100%).

## ETAPE 9 : Cas 2D
- **Validation 2D** : PASS
- **ESS/N** : 0.9980
- **Commentaires** : Comme implémenté, le cas 2D est un produit indépendant de cas 1D (une Gaussienne 2D). L'architecture RealNVP à 6 couches de couplage réussit parfaitement à modéliser cette distribution avec un ESS proche de 1.
- **Instabilités rencontrées** : Aucune instabilité majeure, le gradient clipping à 1.0 (inclus d'office) a parfaitement stabilisé l'apprentissage.

## Semaine 4

### ETAPE 1 : Stress-test non gaussien
- **Validation 1D non gaussien** : PASS
- **ESS/N** : 0.9228
- **Commentaires** : Avec $J(\theta) = 1 + \theta^2$, la cible n'est plus parfaitement gaussienne. Cependant, le RealNVP atteint un ESS/N très solide (>92%). Le pipeline de base est donc robuste et validé avant de passer au MIMO.

### ETAPE 3 : Entraînement et debug MIMO $n_t=2$
- **Validation MIMO $n_t=2$ (d=4)** : PASS
- **ESS/N** : 0.9943
- **Commentaires** : Modèle entraîné sur l'Oracle MIMO CSI Imparfait (cas isotrope). Aucune instabilité numérique rencontrée avec `torch.linalg.slogdet`.

### ETAPE 4 : Extension MIMO $n_t=4$
- **Validation MIMO $n_t=4$ (d=8)** : PASS
- **ESS/N** : 0.9846
- **Commentaires** : Temps d'entraînement rallongé. Le modèle a été configuré avec 10 couches et dim=128. L'ESS/N se maintient remarquablement bien.

### ETAPE 5 : Extension MIMO $n_t=8$
- **Validation MIMO $n_t=8$ (d=16)** : PASS
- **ESS/N** : 0.9885
- **Commentaires** : Flow configuré avec 12 couches de couplage. Très bonne robustesse du RealNVP avec l'oracle MIMO isotrope, l'ESS/N reste >98%.

### Décision J24 (CHECKPOINT HUMAIN)
- **Stabilité de l'entraînement** : Excellente (aucune instabilité grave observée, pas de collapse avec le RealNVP).
- **Temps passé sur le debug** : Minime, car le gradient clipping et l'utilisation de `slogdet` ont suffi à assurer la stabilité.
- **ESS/N final à $n_t=8$** : ~0.9885
- **Décision d'architecture (RealNVP vs NSF)** : Le RealNVP a performé bien au-delà de nos attentes, maintenant un ESS/N proche de 100% même en $d=16$. Par conséquent, nous ne jugeons pas utile de complexifier l'architecture avec des splines (NSF) à ce stade, ce qui évite un coût d'ingénierie et de calcul supplémentaire.
- **Suite suggérée** : Le pipeline Flow est maintenant extrêmement stable. Nous pouvons soit approfondir l'évaluation avale complète de ce modèle pour la livraison, soit démarrer l'implémentation de SVGD (Semaine 5) de manière sûre car la base Oracle et Sampler (`jeffreys_flow`) est consolidée. En attente de validation pour démarrer SVGD.

## Vérification post-J24 : isotrope vs CSI imparfait complet

### 1. Chiffre exact du stress-test non gaussien (ETAPE 1)
- **J(θ) utilisé** : J(θ) = 1 + θ² (non constant, non-gaussien, 1D)
- **ESS/N** : **0.9228** (5000 échantillons, 3000 itérations)
- **λ*** : 0.0792
- **Interprétation** : Légère baisse par rapport au cas gaussien (0.9995), attendue et normale. Le pipeline est robuste.

### 2. Confirmation du type d'oracle utilisé jusqu'ici

L'`oracle_mimo.py` implémentait le **cas isotrope** de la Section III-D du papier Yang & Combes (2025), défini par :
```
J(θ) = [2·γ / denom] · I  +  [4σ⁴ / denom²] · θθᵀ
avec γ = (1 - σ²)/2,  denom = 1 + σ²·‖θ‖²
```
Ce J(θ) est bien **dépendant de θ** (via ‖θ‖²) — c'est donc *déjà* une distribution non-triviale — mais c'est la restriction au sous-cas où la matrice de covariance du canal estimé Γ est proportionnelle à l'identité.

Ce n'est **pas** le cas J constant (J = H_hat^T H_hat / σ² serait constant en θ, ce qui est le cas de CSI parfait). L'oracle isotrope était donc mathématiquement cohérent, mais ne couvrait pas le cas général Γ quelconque.

### 3. Implémentation et résultats CSI imparfait complet

**Nouveau fichier** : `oracle_mimo_csi_full.py`
- Formule générale : J(θ) = (2/denom)·Γ + (4σ⁴/denom²)·θθᵀ, avec Γ arbitraire (d×d, représentation réelle de E[ĥĥᴴ])
- Lambda* résolu par Monte-Carlo (100 000 points), car la formule radiale 1D fermée n'existe que pour le cas isotrope
- Gamma anisotrope de test : valeurs propres [3k, k] avec ratio = 3, puissance totale conservée

**Paramètres run** : n_t=2, d=4, σ²=0.1, ratio_γ=3, 4000 itérations, tag="mimo_nt2_csi_full"
- **λ* (MC)** : 0.101582 (vs 0.1440 isotrope — différence attendue car Γ concentre la puissance différemment)
- **Loss finale** : −4.8499 (convergence stable, sans instabilité)
- **ESS** : 4986.72 / 5000
- **ESS/N** : **0.9973**

### 4. Tableau comparatif n_t=2 (d=4)

| Oracle                            | λ*     | ESS/N  |
|-----------------------------------|--------|--------|
| Isotrope (Γ = γ·I)               | 0.1440 | 0.9943 |
| CSI complet (Γ anisotrope, r=3)  | 0.1016 | 0.9973 |

### 5. Analyse de l'écart

L'écart ESS/N est **infime** (+0.003, dans le bruit d'estimation Monte-Carlo). L'explication : la distribution cible anisotrope est simplement une Gaussienne "étirée" dans certaines directions — une forme que le RealNVP avec couplages alternés capture aussi facilement que la version isotrope. La différence de λ* est plus significative (0.1440 vs 0.1016) car Γ anisotrope redistribue la contrainte de puissance différemment selon les axes.

**Conclusion** : l'oracle CSI complet fonctionne parfaitement (pas d'instabilité numérique grâce à slogdet). Le pipeline Flow est validé sur les deux variantes. Le choix de rester sur RealNVP est pleinement confirmé.

## Test de robustesse — régime fortement non-linéaire

### Motivation
Les ESS/N élevés (~99%) obtenus aux étapes précédentes pourraient refléter un régime où σ²·‖θ‖² ≈ 1.1 (quasi-gaussien) : le dénominateur denom = 1 + σ²·‖θ‖² ≈ 2.1 laisse la FIM proche d'une matrice constante, et la distribution cible reste alors une Gaussienne « légèrement déformée ». Ce test augmente σ² de 0.1 à 0.5 pour atteindre σ²·‖θ‖² ≈ 5.5 (×5), un régime où la non-linéarité de J(θ) domine.

### Paramètres
- **σ²** = 0.5 (vs 0.1 précédemment)
- **P** = A²/9 ≈ 11.11 (identique)
- **Gamma** : anisotrope, ratio=3, Tr(Γ_real) = 1.0
- **λ*** = 0.055996 (MC, 100k points) — plus faible que les tests précédents (0.1016 / 0.1440) car la distribution est plus étalée
- **Flow** : RealNVP, 10 couches, hidden_dim=128, 6000 itérations, lr=5e-4
- **Tag** : `mimo_nt2_csi_full_stress`

### Résultats bruts

| Oracle                              | σ²   | σ²·‖θ‖² eff. | λ*      | Loss finale | ESS/N    |
|-------------------------------------|------|---------------|---------|-------------|----------|
| Isotrope (Γ = γ·I)                 | 0.1  | ~1.11         | 0.1440  | —           | **0.9943** |
| CSI complet (Γ aniso, r=3)         | 0.1  | ~1.11         | 0.1016  | −4.8499     | **0.9973** |
| **CSI complet STRESS (Γ aniso, r=3)** | **0.5** | **~5.86**  | **0.0560** | **−2.9000** | **0.9952** |

### Analyse

1. **Pas de chute d'ESS** : de 0.9973 → 0.9952, soit un écart de −0.2 points, **dans le bruit Monte-Carlo** (N=5000). L'hypothèse d'un ESS artificiellement élevé dû au régime quasi-gaussien est **empiriquement réfutée**.

2. **Le RealNVP est robuste en régime non-linéaire** : malgré un σ²·‖θ‖² ×5, avec denom = 1 + 5.86 ≈ 6.86 (la FIM est écrasée d'un facteur ~7 par rapport à θ=0), le flow maintient un ESS/N > 99.5%. La distribution cible, bien que non-gaussienne, reste unimodale et à queues sub-gaussiennes (grâce au terme −λ*·‖θ‖²), ce qui est dans la zone de confort du RealNVP.

3. **Convergence de la loss** : la loss converge rapidement (plateau dès l'itération ~1000 à −2.90) et reste stable, confirmant que l'optimisation est saine et que le flow capture bien la forme cible.

4. **Explication structurelle** : la FIM J(θ) = 2Γ/denom + 4σ⁴/denom²·θθᵀ varie en ‖θ‖² de manière monotone. Le log-det J est une fonction lisse et concave de ‖θ‖², ce qui, combiné au terme quadratique −λ*·‖θ‖², produit une cible dont les isodensités restent proches d'ellipsoïdes — une forme que le RealNVP avec couplages alternés est structurellement bien adapté à modéliser.

### Conclusion du stress-test

Ce résultat est **très positif** : le RealNVP est confirmé robuste jusqu'à σ²·‖θ‖² ≈ 6 sans aucune dégradation mesurable de l'ESS. L'hypothèse initiale (ESS élevé uniquement dû à un régime quasi-gaussien) est réfutée. Le pipeline Flow est validé pour les régimes opérationnels réalistes. Le passage aux NSF n'est **pas nécessaire** au vu de ces résultats.

**Pistes pour stress-test encore plus extrême** (non implémentées, à discuter au J24) :
1. Augmenter encore σ² vers 0.8–0.9 (σ²·‖θ‖² ≈ 10–20) pour trouver le point de rupture
2. Tester en dimension plus élevée (n_t=8, d=16) combiné avec σ² élevé pour vérifier l'interaction dimensionnalité × non-linéarité
