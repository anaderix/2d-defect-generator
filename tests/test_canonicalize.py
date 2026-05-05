"""Tests for build_group: algebraic structure, action, orbits."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canonicalize import (
    build_coords,
    build_group,
    canonical,
    nearest_neighbours,
)


SIZES = [3, 4, 5, 6, 7]
RNG = np.random.default_rng(0)


def _identity(n):
    return np.arange(2 * n * n)


# -------- (a) algebraic --------

def test_group_size():
    for n in SIZES:
        G = build_group(n)
        assert G.shape == (6 * n * n, 2 * n * n)
        unique = {p.tobytes() for p in G}
        assert len(unique) == 6 * n * n, f"n={n}: dupes in G"


def test_identity_in():
    for n in SIZES:
        G = build_group(n)
        e = _identity(n)
        assert any((p == e).all() for p in G), f"n={n}: identity not in G"


def test_closure():
    for n in SIZES:
        G = build_group(n)
        S = {p.tobytes() for p in G}
        idx = RNG.choice(len(G), size=(500, 2))
        for a_i, b_i in idx:
            a, b = G[a_i], G[b_i]
            composed = a[b]
            assert composed.tobytes() in S, f"n={n}: closure failed"


def test_inverses():
    for n in SIZES:
        G = build_group(n)
        S = {p.tobytes() for p in G}
        for p in G:
            inv = np.argsort(p)
            assert inv.tobytes() in S, f"n={n}: missing inverse"


# -------- (b) D_3 relations --------

def test_c3_order():
    """C₃ at row index n*n*1 + 0 (point op #1, translation 0) should cube to e."""
    for n in SIZES:
        G = build_group(n)
        c3 = G[n * n]  # second point op = c3, translation (0,0)
        cubed = c3[c3[c3]]
        assert (cubed == _identity(n)).all(), f"n={n}: C₃³ ≠ e"


def test_mirror_order():
    for n in SIZES:
        G = build_group(n)
        s = G[3 * n * n]  # 4th point op = sigma, translation (0,0)
        sq = s[s]
        assert (sq == _identity(n)).all(), f"n={n}: σ² ≠ e"


def test_dihedral():
    """(σ·C₃)² = e in D_3."""
    for n in SIZES:
        G = build_group(n)
        c3 = G[n * n]
        s = G[3 * n * n]
        sc = s[c3]
        sq = sc[sc]
        assert (sq == _identity(n)).all(), f"n={n}: (σC₃)² ≠ e"


# -------- (c) action --------

def test_sublattice_preserved():
    for n in SIZES:
        G = build_group(n)
        m = n * n
        for p in G:
            assert (p[:m] < m).all(), f"n={n}: B contaminated by N"
            assert (p[m:] >= m).all(), f"n={n}: N contaminated by B"


def test_distances_preserved():
    for n in SIZES:
        G = build_group(n)
        coords = build_coords(n)
        # Wrap distances on the torus: take min over periodic images
        a1 = np.array([1.0, 0.0])
        a2 = np.array([0.5, np.sqrt(3) / 2])

        def torus_dist(p, q):
            best = np.inf
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    shift = di * n * a1 + dj * n * a2
                    best = min(best, np.linalg.norm(p - q + shift))
            return best

        idx = RNG.choice(len(G), size=20, replace=False)
        for _ in range(50):
            i, j = RNG.integers(0, 2 * n * n, size=2)
            if i == j:
                continue
            d0 = torus_dist(coords[i], coords[j])
            for gi in idx:
                p = G[gi]
                # p is "where to read from"; we need forward action: site k → ?
                # forward[k] = m where perm[m] = k, i.e. forward = argsort(perm)
                fwd = np.argsort(p)
                d1 = torus_dist(coords[fwd[i]], coords[fwd[j]])
                assert np.isclose(d0, d1, atol=1e-9), (
                    f"n={n}: distance changed under group element {gi}"
                )


def test_nn_preserved():
    for n in SIZES:
        G = build_group(n)
        nn = nearest_neighbours(n)
        idx = RNG.choice(len(G), size=15, replace=False)
        for gi in idx:
            fwd = np.argsort(G[gi])
            mapped = {frozenset({fwd[i], fwd[j]}) for pair in nn for i, j in [tuple(pair)]}
            assert mapped == nn, f"n={n}: NN bonds not preserved by g={gi}"


# -------- (d) orbit sanity --------

def test_pure_unique():
    for n in SIZES:
        G = build_group(n)
        t = np.zeros(2 * n * n, dtype=np.int8)
        canons = {canonical(t, G)}
        assert len(canons) == 1


def test_single_CB_orbit():
    for n in SIZES:
        G = build_group(n)
        seen = set()
        for i in range(n * n):
            t = np.zeros(2 * n * n, dtype=np.int8)
            t[i] = 1
            seen.add(canonical(t, G))
        assert len(seen) == 1, f"n={n}: single C_B not unique, got {len(seen)} classes"


def test_single_CN_orbit():
    for n in SIZES:
        G = build_group(n)
        seen = set()
        for i in range(n * n):
            t = np.zeros(2 * n * n, dtype=np.int8)
            t[n * n + i] = 1
            seen.add(canonical(t, G))
        assert len(seen) == 1, f"n={n}: single C_N not unique, got {len(seen)} classes"


def test_orbit_size_at_origin():
    """Single C_B at B(0,0) — orbit must have exactly n² elements."""
    for n in SIZES:
        G = build_group(n)
        t = np.zeros(2 * n * n, dtype=np.int8)
        t[0] = 1
        orbit = {t[p].tobytes() for p in G}
        assert len(orbit) == n * n, f"n={n}: orbit size {len(orbit)} ≠ n²={n*n}"


# -------- (e) canonical invariance --------

def test_canonical_invariant():
    for n in SIZES:
        G = build_group(n)
        for _ in range(50):
            k = int(RNG.integers(0, 5))
            t = np.zeros(2 * n * n, dtype=np.int8)
            sites = RNG.choice(2 * n * n, size=k, replace=False)
            t[sites] = 1
            c0 = canonical(t, G)
            sample = RNG.choice(len(G), size=10, replace=False)
            for gi in sample:
                t_g = t[G[gi]]
                assert canonical(t_g, G) == c0, f"n={n}: canon not invariant under g={gi}"


if __name__ == "__main__":
    import sys
    import traceback

    tests = [
        test_group_size,
        test_identity_in,
        test_closure,
        test_inverses,
        test_c3_order,
        test_mirror_order,
        test_dihedral,
        test_sublattice_preserved,
        test_distances_preserved,
        test_nn_preserved,
        test_pure_unique,
        test_single_CB_orbit,
        test_single_CN_orbit,
        test_orbit_size_at_origin,
        test_canonical_invariant,
    ]

    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}: unexpected exception")
            traceback.print_exc()

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
