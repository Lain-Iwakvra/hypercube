# Hypercube

Projection 2D d'un hypercube *n*-dimensionnel, avec coloration des arêtes par
dimension. L'animation est écrite en vidéo à l'aide de la bibliothèque
partagée [`lib`](https://github.com/Lain-Iwakvra/lib) (`figure_to_bgr`,
`njit`) — le projet dépend donc de `lib`, comme les autres de la collection.

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
pip install numpy matplotlib numba opencv-python tqdm
```

Sur Arch, les bindings Python d'OpenCV viennent de `python-opencv` (paquet
`extra`) : `sudo pacman -S python-opencv`.

Le dossier `lib` doit être voisin de ce projet (le script l'ajoute lui-même au
chemin d'import, aucune variable d'environnement n'est nécessaire).

## Utilisation

```bash
python hypercube.py
```

Le script génère deux vidéos — `hypercube_norme1.mp4` et
`hypercube_fibonacci.mp4` — puis affiche les deux projections côte à côte à
leur état final ; fermez la fenêtre pour quitter. Les images sont écrites
directement en MP4 (via OpenCV), sans fichier intermédiaire.

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
