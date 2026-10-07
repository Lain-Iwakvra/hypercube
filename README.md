# Hypercube

Projection 2D d'un hypercube *n*-dimensionnel, avec coloration des arêtes par
dimension. Projet **autonome** — il ne dépend pas du dépôt `lib`.

## Ce que fait le script

1. **Construction** — l'hypercube `n`-D a `2^n` sommets numérotés de `0` à
   `2^n - 1` ; le bit *k* du numéro indique si la dimension *k* est activée.
   Chaque sommet est projeté en 2D comme la somme des vecteurs de ses
   dimensions actives.
2. **Arêtes** — une arête relie `i` et `i XOR (1 << dim)` : les deux sommets
   ne diffèrent que par une dimension. Le tableau est pré-alloué
   (`n · 2^(n-1)` arêtes) plutôt que rempli par `append`, incompatible avec
   le typage Numba.
3. **Rendu** — chaque arête est colorée selon la dimension qu'elle relie
   (`cm.rainbow`), ce qui rend lisibles les chevauchements.

## Les deux projections comparées

| Gauche — `Norme = 1` | Droite — `Norme = F_n` |
|---|---|
| Tous les vecteurs ont la même longueur : projection isotrope, mais les dimensions hautes se chevauchent beaucoup | Longueurs de Fibonacci (1, 1, 2, 3, 5) : les dimensions basses dominent, les suivantes s'étalent |

## Installation

```bash
pip install numpy matplotlib numba
```

## Utilisation

```bash
python hypercube.py
```

Une fenêtre s'ouvre avec les deux projections côte à côte ; fermez-la pour
quitter. Le temps de calcul est affiché dans le terminal (typiquement
quelques millisecondes — les fonctions `@njit` sont mises en cache).

## Réglages

Dans le bloc `__main__` :

```python
n = 5        # dimension : 5 → 32 sommets et 80 arêtes
             # 6 → 64 sommets / 192 arêtes, 8 → 256 / 1024, ...
```

Attention, le coût mémoire et le nombre d'arêtes explosent en `O(n·2ⁿ)` :
au-delà de `n ≈ 10` le tracé devient illisible.

## Licence

Projet personnel — Paul Fournier.
