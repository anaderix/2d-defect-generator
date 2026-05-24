"""Canonical form for defect placements on hBN n×n monolayer supercell.

Symmetry group p3m1 ⊕ translations, |G| = 6n²:
  - n² translations along lattice vectors a₁, a₂
  - 3-fold rotation C₃ around a B-site
  - 3 mirror planes through B–N bonds

Sublattices are NOT exchanged (B and N are different atoms).

Node index encoding (length 2n²):
  - B(i, j)  →  i*n + j               for i, j ∈ [0, n)
  - N(i, j)  →  n² + i*n + j

Lattice convention: a₁ = (1, 0), a₂ = (1/2, √3/2). N atom in cell (i, j)
sits at offset (1/3, 1/3) fractional relative to B(i, j).
"""

import numpy as np


def _site_idx(n: int, c: int, i: int, j: int) -> int:
    return c * n * n + (i % n) * n + (j % n)


def _point_ops():
    """Return list of 6 functions (c, i, j) → (c', i', j') in fractional lattice
    coordinates, for the point group 3m around B(0, 0).

    Lattice: a₁ = (1, 0), a₂ = (-1/2, √3/2), δ_BN = (1/3)a₁ + (-1/3)a₂
    (matches BN.in convention).

    Derivation:
      C₃·a₁ = a₂, C₃·a₂ = -a₁ - a₂   ⇒  on B: (i, j) → (-j, i - j)
      C₃·δ_BN = (1/3)a₁ + (2/3)a₂    ⇒  N-channel needs +1 shift on j
                                         ⇒  on N: (i, j) → (-j, i - j + 1)
      σ (mirror through B(0,0)–N(0,0) bond): a₁ → -a₂, a₂ → -a₁
                          ⇒  on both channels: (i, j) → (-j, -i)
    """

    def e(c, i, j):
        return c, i, j

    def c3(c, i, j):
        if c == 0:
            return c, -j, i - j
        return c, -j, i - j + 1

    def c3_sq(c, i, j):
        return c3(*c3(c, i, j))

    def sigma(c, i, j):
        return c, -j, -i

    def sigma_c3(c, i, j):
        return sigma(*c3(c, i, j))

    def sigma_c3_sq(c, i, j):
        return sigma(*c3_sq(c, i, j))

    return [e, c3, c3_sq, sigma, sigma_c3, sigma_c3_sq]


def build_group(n: int) -> np.ndarray:
    """Return array of shape (6n², 2n²) of permutations.

    For each group element g, perm[g] satisfies:
        new_tensor = old_tensor[perm[g]]
    i.e. perm[g][m] is the source index in the original tensor whose value
    ends up at position m after applying g.
    """
    point_ops = _point_ops()
    nodes = 2 * n * n
    perms = np.zeros((6 * n * n, nodes), dtype=np.int64)
    row = 0
    for op in point_ops:
        for ti in range(n):
            for tj in range(n):
                perm = np.zeros(nodes, dtype=np.int64)
                for c in range(2):
                    for i in range(n):
                        for j in range(n):
                            src = _site_idx(n, c, i, j)
                            nc, ni, nj = op(c, i, j)
                            ni += ti
                            nj += tj
                            dst = _site_idx(n, nc, ni, nj)
                            perm[dst] = src
                perms[row] = perm
                row += 1
    return perms


def canonical(tensor_flat: np.ndarray, perms: np.ndarray) -> bytes:
    """Lex-min over the orbit. tensor_flat must be 1-D length 2n²."""
    return min(tensor_flat[p].tobytes() for p in perms)


def build_coords(n: int) -> np.ndarray:
    """Cartesian coordinates of all 2n² sites, shape (2n², 2). For tests."""
    a1 = np.array([1.0, 0.0])
    a2 = np.array([-0.5, np.sqrt(3) / 2])
    delta_N = a1 / 3.0 - a2 / 3.0
    coords = np.zeros((2 * n * n, 2))
    for c in range(2):
        for i in range(n):
            for j in range(n):
                pos = i * a1 + j * a2 + (delta_N if c == 1 else np.zeros(2))
                coords[_site_idx(n, c, i, j)] = pos
    return coords


def make_benzene(n: int, i: int = 0, j: int = 0) -> np.ndarray:
    """Benzene ring centred between cells (i, j) and (i+1, j+1).

    Pattern: B(i,j), B(i+1,j), B(i+1,j+1), N(i,j), N(i,j+1), N(i+1,j+1).
    Returns 1-D tensor of length 2n².
    """
    t = np.zeros(2 * n * n, dtype=np.int8)
    for di, dj in [(0, 0), (1, 0), (1, 1)]:
        t[_site_idx(n, 0, i + di, j + dj)] = 1  # B
    for di, dj in [(0, 0), (0, 1), (1, 1)]:
        t[_site_idx(n, 1, i + di, j + dj)] = 1  # N
    return t


def make_naphthalene(n: int, i: int = 0, j: int = 0) -> np.ndarray:
    """Naphthalene = two fused benzene rings (along one lattice direction).

    Extracted from maciej_BN/BN-5-5-1-napthalene anchored at origin:
      B: (i, j), (i+1, j), (i+1, j+1), (i+2, j), (i+2, j+1)
      N: (i, j), (i, j+1), (i+1, j), (i+1, j+1), (i+2, j+1)
    """
    t = np.zeros(2 * n * n, dtype=np.int8)
    for di, dj in [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)]:
        t[_site_idx(n, 0, i + di, j + dj)] = 1
    for di, dj in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]:
        t[_site_idx(n, 1, i + di, j + dj)] = 1
    return t


def enumerate_motifs(n: int, k_B: int, k_N: int, perms: np.ndarray | None = None) -> dict:
    """All unique canonical configurations with k_B C_B defects and k_N C_N defects.

    Returns dict: canonical_bytes → representative tensor (1-D, length 2n²).
    """
    from itertools import combinations

    if perms is None:
        perms = build_group(n)
    nodes = 2 * n * n
    m = n * n
    classes: dict[bytes, np.ndarray] = {}
    for B_sites in combinations(range(m), k_B):
        for N_sites in combinations(range(m), k_N):
            t = np.zeros(nodes, dtype=np.int8)
            for s in B_sites:
                t[s] = 1
            for s in N_sites:
                t[m + s] = 1
            c = canonical(t, perms)
            if c not in classes:
                classes[c] = t.copy()
    return classes


def count_classes(n: int, k: int) -> dict:
    """For each (k_B, k_N) with k_B + k_N = k, count unique classes."""
    perms = build_group(n)
    out = {}
    for k_B in range(k + 1):
        k_N = k - k_B
        out[(k_B, k_N)] = len(enumerate_motifs(n, k_B, k_N, perms))
    return out


def _ring_canons(n: int, perms: np.ndarray) -> tuple[bytes, bytes]:
    """Cache-friendly: canonical forms of benzene and napthalene at origin."""
    return (
        canonical(make_benzene(n, 0, 0), perms),
        canonical(make_naphthalene(n, 0, 0), perms),
    )


def summary_label(tensor: np.ndarray, n: int, perms: np.ndarray | None = None) -> str:
    """Human-readable label for a placement, compatible with maciej_BN/ naming
    where possible. Falls back to a stable hash for novel configurations.

    Output examples:
      'pure'                 — no defects
      'C_B' / 'C_N'          — single substitution
      'benzene'              — 6-atom ring (3 C_B + 3 C_N hexagon)
      'napthalene'           — 10-atom fused two-ring
      'cb2cn0_a3f9c2'        — generic: counts + 6-hex hash
    """
    if perms is None:
        perms = build_group(n)
    m = n * n
    k_B = int(tensor[:m].sum())
    k_N = int(tensor[m:].sum())

    if k_B == 0 and k_N == 0:
        return "pure"
    if k_B == 1 and k_N == 0:
        return "C_B"
    if k_B == 0 and k_N == 1:
        return "C_N"

    canon = canonical(tensor, perms)
    benzene_c, naph_c = _ring_canons(n, perms)
    if canon == benzene_c:
        return "benzene"
    if canon == naph_c:
        return "napthalene"

    import hashlib
    h = hashlib.sha256(canon).hexdigest()[:6]
    return f"cb{k_B}cn{k_N}_{h}"


def motif_dirname(tensor: np.ndarray, n: int, perms: np.ndarray | None = None) -> str:
    """Full directory name compatible with the BN-{n}-{n}-{layers}-{defect} scheme."""
    return f"BN-{n}-{n}-1-{summary_label(tensor, n, perms)}"


def nearest_neighbours(n: int) -> set:
    """Set of frozenset({i, j}) for all B–N nearest-neighbour pairs (within PBC).

    Lattice: a₂ = (-1/2, √3/2), δ_BN = (1/3)a₁ - (1/3)a₂. The three N atoms
    at bond-length from B(i,j) are N(i,j), N(i-1,j), N(i,j+1).
    """
    pairs = set()
    for i in range(n):
        for j in range(n):
            b = _site_idx(n, 0, i, j)
            for di, dj in [(0, 0), (-1, 0), (0, 1)]:
                nn = _site_idx(n, 1, i + di, j + dj)
                pairs.add(frozenset({b, nn}))
    return pairs
