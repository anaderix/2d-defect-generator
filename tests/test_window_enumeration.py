"""Tests for the window-mode enumerator (algorithm="window").

Variant (B): two configs inside the n×n corner window of an m×m host are
equivalent iff some g ∈ G(m) maps one to the other and keeps the support
inside the window. Hash is computed on the 2n² window slice and is *not*
stable across m, because m×m PBC differs from n×n PBC.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canonicalize import build_group, enumerate_motifs
from generate_geometry import (
    enumerate_window_motifs,
    window_indices,
)


def test_window_indices_shape_and_mapping():
    n, m = 3, 7
    w = window_indices(n, m)
    assert w.shape == (2 * n * n,)
    # B-channel indices: i*m + j for (i, j) ∈ [0,n)²
    for i in range(n):
        for j in range(n):
            assert w[i * n + j] == i * m + j
            assert w[n * n + i * n + j] == m * m + i * m + j
    # All indices unique and in [0, 2m²).
    assert len(set(w.tolist())) == 2 * n * n
    assert int(w.max()) < 2 * m * m
    print("ok  test_window_indices_shape_and_mapping")


def test_window_pure_one_class():
    """k = 0 (no defects) → exactly one class (empty config)."""
    for (n, m) in [(3, 3), (3, 5), (3, 7), (4, 8)]:
        cls = enumerate_window_motifs(n, m, 0, 0)
        assert len(cls) == 1, f"expected 1 pure class, got {len(cls)} for n={n}, m={m}"
        # The representative tensor is all zeros and length 2n².
        ((_, t),) = cls.items()
        assert t.shape == (2 * n * n,)
        assert int(t.sum()) == 0
    print("ok  test_window_pure_one_class")


def test_window_representative_inside_window():
    """Every stored representative is a 2n² window tensor (by construction)."""
    n, m = 3, 7
    for k_B, k_N in [(1, 0), (0, 1), (1, 1), (2, 1)]:
        cls = enumerate_window_motifs(n, m, k_B, k_N)
        for t in cls.values():
            assert t.shape == (2 * n * n,), f"representative wrong shape: {t.shape}"
            assert int(t.sum()) == k_B + k_N
    print("ok  test_window_representative_inside_window")


def test_window_matches_canonical_when_m_equals_n():
    """When m == n, the window covers the entire cell and no support can
    escape; G(m) = G(n) → window-mode classes correspond 1:1 with canonical
    classes from enumerate_motifs."""
    n = 3
    for k_B, k_N in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)]:
        cls_w = enumerate_window_motifs(n, n, k_B, k_N)
        cls_c = enumerate_motifs(n, k_B, k_N)
        assert len(cls_w) == len(cls_c), (
            f"window={len(cls_w)} vs canonical={len(cls_c)} for n=m={n}, "
            f"k_B={k_B}, k_N={k_N}"
        )
    print("ok  test_window_matches_canonical_when_m_equals_n")


def test_window_produces_strictly_more_when_m_greater_than_n():
    """Core property: when m > n, window-mode dedup is coarser ⇒ produces
    at least as many classes as canonical, and strictly more for any (k_B, k_N)
    where the n×n translational subgroup actually identified distinct
    placements (which is essentially every non-trivial case)."""
    n, m = 3, 7
    found_strict = False
    for k_B, k_N in [(1, 0), (0, 1), (2, 0), (1, 1), (2, 1), (1, 2)]:
        cls_w = enumerate_window_motifs(n, m, k_B, k_N)
        cls_c = enumerate_motifs(n, k_B, k_N)
        assert len(cls_w) >= len(cls_c), (
            f"window-mode produced fewer classes ({len(cls_w)}) than canonical "
            f"({len(cls_c)}) for (n={n}, m={m}, k_B={k_B}, k_N={k_N}) — "
            f"window dedup must be at least as coarse"
        )
        if len(cls_w) > len(cls_c):
            found_strict = True
            print(f"      (k_B={k_B}, k_N={k_N}): canonical={len(cls_c)}, "
                  f"window={len(cls_w)}  +{len(cls_w) - len(cls_c)}")
    assert found_strict, (
        "expected at least one (k_B, k_N) where window > canonical "
        "for m > n; n×n translations are broken so this should never tie"
    )
    print("ok  test_window_produces_strictly_more_when_m_greater_than_n")


def test_window_single_defect_collapses_to_one_class():
    """A single defect: G(m) translations can carry it from any window cell
    to any other window cell (endpoints both inside the window), so under
    variant (B) all n² placements collapse to one class. Same for C_B and C_N."""
    n, m = 3, 7
    cls_b = enumerate_window_motifs(n, m, 1, 0)
    cls_n = enumerate_window_motifs(n, m, 0, 1)
    assert len(cls_b) == 1, f"single-C_B should collapse to 1 class, got {len(cls_b)}"
    assert len(cls_n) == 1, f"single-C_N should collapse to 1 class, got {len(cls_n)}"
    print("ok  test_window_single_defect_collapses_to_one_class")


TESTS = [
    test_window_indices_shape_and_mapping,
    test_window_pure_one_class,
    test_window_representative_inside_window,
    test_window_matches_canonical_when_m_equals_n,
    test_window_produces_strictly_more_when_m_greater_than_n,
    test_window_single_defect_collapses_to_one_class,
]


def main() -> int:
    passed = failed = 0
    for t in TESTS:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
