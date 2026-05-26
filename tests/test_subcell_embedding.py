"""Sanity tests for the sub-supercell-in-host embedding logic.

Verifies that enumerating motifs on a small N×N and embedding them in a
larger M×M host doesn't introduce spurious collapse or over-counting.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canonicalize import build_group, canonical, enumerate_motifs
from generate_geometry import embed_tensor, motif_dirname


def test_embed_identity_when_m_equals_n():
    """embed_tensor(t, n, n) returns a copy of t."""
    n = 3
    classes = enumerate_motifs(n, 1, 1)
    for t in classes.values():
        out = embed_tensor(t, n, n)
        assert np.array_equal(out, t), "embed with m==n should be identity"
    print("ok  test_embed_identity_when_m_equals_n")


def test_embed_pristine_stays_pristine():
    """Embedding the all-zeros tensor stays all-zeros at any host size."""
    n, m = 2, 5
    t_sub = np.zeros(2 * n * n, dtype=np.int8)
    t_full = embed_tensor(t_sub, n, m)
    assert t_full.shape == (2 * m * m,)
    assert (t_full == 0).all()
    print("ok  test_embed_pristine_stays_pristine")


def test_embed_corner_anchored():
    """A C at (i,j) in n×n lands at (i,j) in m×m, with (i,j) ∈ [0, n)²."""
    n, m = 2, 4
    t_sub = np.zeros(2 * n * n, dtype=np.int8)
    t_sub[1 * n + 0] = 1            # C_B at (1, 0)
    t_sub[n * n + 0 * n + 1] = 1    # C_N at (0, 1)
    t_full = embed_tensor(t_sub, n, m)
    # In the m×m tensor: C_B at (1, 0) → t_full[1*m + 0] = t_full[4]
    assert t_full[1 * m + 0] == 1, "C_B should land at (1, 0) in host"
    assert t_full[m * m + 0 * m + 1] == 1, "C_N should land at (0, 1) in host"
    # And nothing else
    assert int(t_full.sum()) == 2
    print("ok  test_embed_corner_anchored")


def test_no_spurious_collapse_in_host():
    """Distinct n×n orbit representatives stay distinct after embedding,
    measured by their canonical form under the m×m symmetry group.

    Uses n=3, m=5 so the test is fast but non-trivial (n=3 (1,1) gives 3
    classes; verify all 3 embedded placements canonicalize differently
    under the m=5 group).
    """
    n, m = 3, 5
    perms_n = build_group(n)
    perms_m = build_group(m)
    classes_n = enumerate_motifs(n, 1, 1, perms_n)

    host_canons = set()
    for t_sub in classes_n.values():
        t_full = embed_tensor(t_sub, n, m)
        host_canons.add(canonical(t_full, perms_m))

    assert len(host_canons) == len(classes_n), (
        f"expected {len(classes_n)} distinct host orbits, "
        f"got {len(host_canons)} — embedding caused spurious collapse"
    )
    print(f"ok  test_no_spurious_collapse_in_host  ({len(classes_n)} classes preserved)")


def test_m_less_than_n_raises():
    """embed_tensor should refuse m < n."""
    n = 4
    t = np.zeros(2 * n * n, dtype=np.int8)
    try:
        embed_tensor(t, n, m=3)
    except ValueError:
        print("ok  test_m_less_than_n_raises")
        return
    raise AssertionError("expected ValueError for m < n")


def test_motif_dirname_rules():
    """Directory naming follows the agreed rules."""
    cases = [
        # (n, m, k_B, k_N, label, expected)
        (3, 3, 0, 0, "pure",         "BN-3-3-1-pure"),
        (3, 8, 0, 0, "pure",         "BN-8-8-1-pure"),
        (3, 3, 1, 0, "C_B",          "BN-3-3-1-C_B"),
        (3, 8, 1, 0, "C_B",          "BN-8-8-1-N3-C_B"),
        (4, 4, 2, 0, "cb2cn0_1f44e1","BN-4-4-1-cb2cn0_1f44e1"),
        (4, 8, 2, 0, "cb2cn0_1f44e1","BN-8-8-1-N4-cb2cn0_1f44e1"),
    ]
    for n, m, k_B, k_N, label, expected in cases:
        got = motif_dirname(n, m, k_B, k_N, label)
        assert got == expected, f"motif_dirname({n},{m},{k_B},{k_N},{label!r}) → {got!r}, expected {expected!r}"
    print(f"ok  test_motif_dirname_rules  ({len(cases)} cases)")


TESTS = [
    test_embed_identity_when_m_equals_n,
    test_embed_pristine_stays_pristine,
    test_embed_corner_anchored,
    test_no_spurious_collapse_in_host,
    test_m_less_than_n_raises,
    test_motif_dirname_rules,
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
