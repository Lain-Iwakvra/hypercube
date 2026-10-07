"""hypercube.py — projection 2D d'un hypercube n-dimensionnel.

Construit le graphe complet d'un hypercube `n`-D (`2^n` sommets,
`n·2^(n-1)` arêtes), le projette dans le plan en attribuant à chaque
dimension un vecteur propre, puis dessine chaque arête colorée selon la
dimension qu'elle relie.

Deux projections sont comparées côte à côte :
    • Norme = 1    : tous les vecteurs de dimension ont la même longueur
    • Norme = F_n  : longueurs de Fibonacci (1, 1, 2, 3, 5), ce qui rend
                     les dimensions basses dominantes et « étale » le reste

Dépendances : numpy, matplotlib, numba (obligatoire)

Usage :
    python hypercube.py      # puis fermer la fenêtre pour quitter
Réglages : variable `n` dans le bloc `__main__` (n = 5 → 32 sommets, 80 arêtes).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numba import njit
import time


@njit(cache=True, fastmath=True)
def build_hypercube(n, vectors):
    """Construit sommets et arêtes d'un hypercube n-D projeté en 2D.

    Sommets : le bit k du numéro `i` vaut 1 ⟺ la dimension k est activée ;
              le sommet i est donc la somme des vecteurs de dimensions activées.
    Arêtes  : j = i XOR (1 << dim) — on ne garde que i < j pour ne pas
              compter chaque arête deux fois.

    Retourne (vertices, edges) où vertices est (2^n, 2) et edges (n·2^(n-1), 2).
    """
    num_vertices = 1 << n
    vertices = np.zeros((num_vertices, 2))
    for i in range(num_vertices):
        for dim in range(n):
            if (i >> dim) & 1:
                vertices[i, 0] += vectors[dim, 0]
                vertices[i, 1] += vectors[dim, 1]
    # Chaque sommet a exactement n voisins ; chaque arête est comptée 2x.
    # On pré-alloue donc le tableau au lieu de faire des append dans numba.
    edge_count = n * num_vertices // 2
    edges = np.empty((edge_count, 2), dtype=np.int64)
    k = 0
    for i in range(num_vertices):
        for dim in range(n):
            j = i ^ (1 << dim)
            if i < j:
                edges[k, 0] = i
                edges[k, 1] = j
                k += 1
    return vertices, edges


def fixed_angle_vectors(lengths):
    """Vecteurs de projection : n directions régulièrement réparties sur [0, π].

    Toujours en NumPy car trigonométrie → NumPy est optimal ici.
    """
    n = len(lengths)
    if n == 0:
        return np.zeros((0, 2))
    k = np.arange(n)
    angles = np.pi * k / n
    return np.column_stack((lengths * np.cos(angles), lengths * np.sin(angles)))


@njit(cache=True)
def edges_dimension_mapping(edges):
    """Pour chaque arête, retourne la dimension qu'elle relie.

    Les deux extrémités diffèrent exactement d'un bit : `a XOR b` est donc
    une puissance de 2, et le rang de ce bit est la dimension.
    (équivalent de int(log2(diff)) sans appel à log)
    """
    m = edges.shape[0]
    dims = np.empty(m, dtype=np.int64)
    for i in range(m):
        a, b = edges[i, 0], edges[i, 1]
        diff = a ^ b   # toujours une puissance de 2
        d = 0
        while diff > 1:   # équivalent à int(log2(diff))
            diff >>= 1
            d += 1
        dims[i] = d
    return dims


def fibonacci_lengths(n):
    """n premiers termes de Fibonacci (1, 1, 2, 3, 5, …) en float64."""
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    fib = np.zeros(n, dtype=np.float64)
    fib[0] = 1
    if n > 1:
        fib[1] = 1
        for i in range(2, n):
            fib[i] = fib[i - 1] + fib[i - 2]
    return fib


def plot_hypercube(ax, n, lengths, title):
    """Dessine l'hypercube projeté sur l'Axe `ax`, une couleur par dimension."""
    vectors = fixed_angle_vectors(lengths)
    vertices, edges = build_hypercube(n, vectors)
    edge_dims = edges_dimension_mapping(edges)
    colors = cm.rainbow(np.linspace(0, 1, n))

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12)

    for (a, b), dim in zip(edges, edge_dims):
        ax.plot([vertices[a, 0], vertices[b, 0]],
                [vertices[a, 1], vertices[b, 1]],
                color=colors[dim], lw=1.2, alpha=0.9)


if __name__ == "__main__":
    t_start = time.time()
    n = 5                      # dimension de l'hypercube (5 → 32 sommets)
    lengths_ones = np.ones(n)  # projection isotrope
    lengths_fibo = fibonacci_lengths(n)   # projection « écrasée » en bas

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    plot_hypercube(axes[0], n, lengths_ones, "Norme = 1")
    plot_hypercube(axes[1], n, lengths_fibo, "Norme = F_n")

    print(f"Temps écoulé : {time.time()-t_start:.4f} s")
    plt.show()

