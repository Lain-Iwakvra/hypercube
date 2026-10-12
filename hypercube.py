"""hypercube.py — projection 2D d'un hypercube n-dimensionnel.

Construit le graphe complet d'un hypercube `n`-D (`2^n` sommets,
`n·2^(n-1)` arêtes), le projette dans le plan en attribuant à chaque
dimension un vecteur propre, puis dessine chaque arête colorée selon la
dimension qu'elle relie.

Deux projections sont produites, chacune enregistrée dans une vidéo :
    • Norme = 1    : tous les vecteurs de dimension ont la même longueur
    • Norme = F_n  : longueurs de Fibonacci (1, 1, 2, 3, 5), ce qui rend
                     les dimensions basses dominantes et « étale » le reste

Chaque vidéo est une animation de la dimension 0 à la dimension n : la
longueur (norme) de chaque dimension croît de 0 jusqu'à sa valeur finale.
On voit donc successivement un point s'étirer en ligne, puis la ligne
balayer pour former un carré, le carré engendrer un cube, puis un tesseract…

L'écriture vidéo et l'accélération JIT viennent de la bibliothèque partagée
`lib` (`figure_to_bgr`, `njit`), comme les autres projets de la collection.

Dépendances : numpy, matplotlib, opencv-python, tqdm, **lib**
    (lib fournit `figure_to_bgr` et un `njit` avec repli si numba est absent)

Le dossier « lib » est supposé voisin de ce script (le script l'ajoute lui-même
au chemin d'import).

Usage :
    python hypercube.py      # génère hypercube_norme1.mp4 et hypercube_fibonacci.mp4
Réglages : variable `n` dans le bloc `__main__` (n = 5 → 32 sommets, 80 arêtes).
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from tqdm import tqdm

# Le dossier « lib » est toujours voisin du dossier de ce script : on l'ajoute
# au chemin d'import pour rendre `lib` disponible sans configuration.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
from lib import figure_to_bgr, njit


@njit(cache=True, fastmath=True)
def vertices_from_vectors(n, vectors):
    """Position des 2^n sommets : somme des vecteurs des dimensions activées.

    Le bit k du numéro `i` vaut 1 ⟺ la dimension k est activée ; le sommet i
    est donc la somme des vecteurs de dimensions activées. Des longueurs
    nulles font coïncider des sommets (l'hypercube dégénère), ce qui est
    justement exploité par l'animation.
    """
    num_vertices = 1 << n
    vertices = np.zeros((num_vertices, 2))
    for i in range(num_vertices):
        for dim in range(n):
            if (i >> dim) & 1:
                vertices[i, 0] += vectors[dim, 0]
                vertices[i, 1] += vectors[dim, 1]
    return vertices


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
    vertices = vertices_from_vectors(n, vectors)
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


def animated_lengths(n, target, progress):
    """Longueurs des dimensions pour un avancement `progress` dans [0, n].

    Les dimensions déjà franchies ont leur longueur finale, la dimension en
    cours croît de 0 à sa longueur finale, et les suivantes restent à 0 (donc
    invisibles) : l'hypercube dégénère alors à la dimension courante, ce qui
    donne successivement point → ligne → carré → cube → tesseract…
    """
    lengths = np.zeros(n, dtype=np.float64)
    d = int(progress)
    if d >= n:
        return np.asarray(target, dtype=np.float64).copy()
    lengths[:d] = target[:d]
    lengths[d] = (progress - d) * target[d]
    return lengths


def make_animation(n, target, path, frames_per_dim=60, fps=30,
                   hold_start=0.5, hold_end=1.0):
    """Génère la vidéo `path` : de la dimension 0 à n, normes croissantes."""
    target = np.asarray(target, dtype=np.float64)

    # Topologie et couleurs fixes : seule la géométrie (sommets) varie.
    vectors_full = fixed_angle_vectors(target)
    _, edges = build_hypercube(n, vectors_full)
    edge_dims = edges_dimension_mapping(edges)
    colors = cm.rainbow(np.linspace(0, 1, n))

    # Le cadre suivra l'objet à chaque image. On déduit seulement une taille
    # minimale du cadre (15 % de l'étendue finale) pour que le point de départ
    # reste bien visible sans zoomer à l'infini.
    v_full = vertices_from_vectors(n, vectors_full)
    final_half = 0.5 * max(np.ptp(v_full[:, 0]), np.ptp(v_full[:, 1]))
    min_half = 0.15 * final_half if final_half > 0 else 1.0

    figsize, dpi = (8, 8), 150          # 8x8 pouces @150 dpi → images 1200x1200
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(-min_half, min_half)
    ax.set_ylim(-min_half, min_half)

    lines = [ax.plot([], [], color=colors[d], lw=1.2, alpha=0.9)[0]
             for d in edge_dims]

    # Vitesse d'agrandissement constante : chaque dimension gagne le même
    # nombre d'unités de longueur par image. Une grande dimension met donc
    # proportionnellement plus de temps. On normalise pour garder la même
    # durée totale (n * frames_per_dim images de croissance).
    total = float(target.sum())
    frames_per_unit = (n * frames_per_dim / total) if total > 0 else 0.0

    # Avancements : pause sur le point, croissance dimension par dimension,
    # puis pause sur l'hypercube complet.
    progress_values = [0.0] * max(1, int(round(hold_start * fps)))
    for d in range(n):
        frames_d = max(1, int(round(target[d] * frames_per_unit)))
        for k in range(1, frames_d + 1):
            progress_values.append(d + k / frames_d)
    progress_values += [float(n)] * max(1, int(round(hold_end * fps)))

    def render(prog):
        """Met à jour la figure pour l'avancement `prog` (cadre recadré)."""
        lengths = animated_lengths(n, target, prog)
        vertices = vertices_from_vectors(n, fixed_angle_vectors(lengths))
        for line, (a, b) in zip(lines, edges):
            line.set_data([vertices[a, 0], vertices[b, 0]],
                          [vertices[a, 1], vertices[b, 1]])

        # Cadre recadré sur l'objet à l'instant courant : il zoom/dézoom pour
        # garder l'objet bien cadré tout au long de la croissance.
        (x0, y0), (x1, y1) = vertices.min(axis=0), vertices.max(axis=0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = 1.08 * max((x1 - x0) / 2, (y1 - y0) / 2, min_half)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)

    # Écriture directe en MP4 via OpenCV : chaque image est capturée avec
    # lib.figure_to_bgr, aucune image intermédiaire sur le disque.
    size = (int(figsize[0] * dpi), int(figsize[1] * dpi))
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size,
                             isColor=True)
    if not writer.isOpened():
        raise RuntimeError("Impossible d'ouvrir le flux vidéo (codec mp4v)")

    for prog in tqdm(progress_values, desc=os.path.basename(path), unit="img"):
        render(prog)
        writer.write(figure_to_bgr(fig))

    writer.release()
    plt.close(fig)
    return path


if __name__ == "__main__":
    t_start = time.time()
    n = 7                      # dimension de l'hypercube
    out_dir = os.path.dirname(os.path.abspath(__file__))

    suites = [
        ("hypercube_norme1", np.ones(n), "Norme = 1"),
        ("hypercube_fibonacci", fibonacci_lengths(n), "Norme = F_n"),
    ]

    for filename, lengths, _ in suites:
        path = os.path.join(out_dir, filename + ".mp4")
        saved = make_animation(n, lengths, path)
        print(f"Vidéo générée : {saved}")

    print(f"Temps écoulé : {time.time()-t_start:.4f} s")

    # État final côte à côte, comme dans le script d'origine.
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (_, lengths, label) in zip(axes, suites):
        plot_hypercube(ax, n, lengths, label)
    plt.show()

