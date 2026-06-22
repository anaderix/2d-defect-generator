"""Sanity tests for the sub-supercell-in-host embedding logic.

Verifies that enumerating motifs on a small N×N and embedding them in a
larger M×M host doesn't introduce spurious collapse or over-counting.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canonicalize import (
    build_coords,
    build_group,
    canonical,
    enumerate_motifs,
    summary_label,
)
from generate_geometry import (
    _compactness_score,
    choose_compact_representative,
    embed_tensor,
    motif_dirname,
)


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


def _cbcn3_wrap(n: int) -> np.ndarray:
    """CB(CN)3 tetramer placed so it wraps the n×n j-boundary.

    The 3 nearest N neighbours of B(i, j) on this lattice convention are
    N(i, j), N(i-1, j), and N(i, j+1). Anchoring B at (0, n-1) makes the
    third neighbour wrap: N(0, n) ≡ N(0, 0).
    """
    nsq = n * n
    t = np.zeros(2 * nsq, dtype=np.int8)
    t[0 * n + (n - 1)] = 1             # C_B at (0, n-1)
    t[nsq + 0 * n + (n - 1)] = 1       # C_N at (0, n-1)
    t[nsq + (n - 1) * n + (n - 1)] = 1 # C_N at (n-1, n-1)  (wraps in i)
    t[nsq + 0 * n + 0] = 1             # C_N at (0, 0)       (wraps in j)
    return t


def test_embed_centered_offset():
    """centered=True places the n×n block at offset ((m-n)//2, (m-n)//2)."""
    n, m = 3, 7
    off = (m - n) // 2  # = 2
    t_sub = np.zeros(2 * n * n, dtype=np.int8)
    t_sub[1 * n + 0] = 1            # C_B at (1, 0) in n×n
    t_sub[n * n + 0 * n + 2] = 1    # C_N at (0, 2) in n×n
    t_full = embed_tensor(t_sub, n, m, centered=True)
    assert t_full[(1 + off) * m + (0 + off)] == 1, "C_B should land at centred offset"
    assert t_full[m * m + (0 + off) * m + (2 + off)] == 1, "C_N should land at centred offset"
    assert int(t_full.sum()) == 2
    print("ok  test_embed_centered_offset")


def test_compact_representative_preserves_canonical_class():
    """choose_compact_representative must stay inside the orbit, so the
    canonical hash and summary_label are invariant."""
    n = 4
    perms = build_group(n)
    for k_B, k_N in [(1, 1), (2, 0), (2, 1), (1, 3)]:
        for t in enumerate_motifs(n, k_B, k_N, perms).values():
            t_rep = choose_compact_representative(t, n, perms)
            assert canonical(t_rep, perms) == canonical(t, perms), (
                f"compact representative left the orbit for (k_B={k_B}, k_N={k_N})"
            )
            assert summary_label(t_rep, n, perms) == summary_label(t, n, perms)
    print("ok  test_compact_representative_preserves_canonical_class")


def test_compact_representative_unwraps_boundary_motif():
    """The CB(CN)3 wrap motif must be reassembled: max pairwise (flat)
    distance after representative selection should drop below the n×n
    diagonal — confirming the four atoms no longer sit at opposite corners."""
    n = 3
    perms = build_group(n)
    t_wrap = _cbcn3_wrap(n)

    score_before = _compactness_score(t_wrap, n)
    t_rep = choose_compact_representative(t_wrap, n, perms)
    score_after = _compactness_score(t_rep, n)

    assert score_after <= score_before, "representative must not be looser"
    # The CB(CN)3 tetramer fits inside a 2-cell radius; max bond length is
    # √(1/3) ≈ 0.577 and the farthest pair of NN-Ns is < 1.0 in lattice units.
    # The wrap-around flat layout has pairs ~2 cells apart.
    assert score_after[0] < 1.5, (
        f"compact representative still split across boundary: max dist {score_after[0]:.3f}"
    )
    assert score_before[0] > score_after[0], (
        f"score did not improve: before={score_before[0]:.3f} after={score_after[0]:.3f}"
    )
    print(f"ok  test_compact_representative_unwraps_boundary_motif  "
          f"(max pair dist {score_before[0]:.2f} → {score_after[0]:.2f})")


def test_centered_embedding_keeps_cbcn3_connected_in_host():
    """End-to-end colleague scenario: a wrap-around CB(CN)3 enumerated on
    n=3, embedded into m=7 via compact-representative + centred placement,
    must land with all four C atoms within a small flat-space radius — i.e.
    not split by the m×m PBC.
    """
    n, m = 3, 7
    perms = build_group(n)
    t_wrap = _cbcn3_wrap(n)
    t_rep = choose_compact_representative(t_wrap, n, perms)
    t_full = embed_tensor(t_rep, n, m, centered=True)

    # Use build_coords(m) to measure flat pairwise distances of defect atoms
    # in the host. No PBC wrapping applied: if any pair is farther than the
    # n×n diagonal (~n in lattice units), the tetramer got fragmented.
    inds = np.nonzero(t_full)[0]
    coords_m = build_coords(m)[inds]
    max_pair = 0.0
    for a in range(len(coords_m)):
        for b in range(a + 1, len(coords_m)):
            max_pair = max(max_pair, float(np.linalg.norm(coords_m[a] - coords_m[b])))

    assert max_pair < 1.5, (
        f"embedded CB(CN)3 is split in the m×m host (max pair dist {max_pair:.2f}); "
        f"corner-anchored embedding would have fragmented it"
    )
    # And the atoms should sit near the host centre, not at corner (0,0).
    centre = coords_m.mean(axis=0)
    host_centre = build_coords(m).mean(axis=0)
    assert np.linalg.norm(centre - host_centre) < n, (
        f"defects not near host centre: {centre} vs host centre {host_centre}"
    )
    print(f"ok  test_centered_embedding_keeps_cbcn3_connected_in_host  "
          f"(max pair dist {max_pair:.2f})")


TESTS = [
    test_embed_identity_when_m_equals_n,
    test_embed_pristine_stays_pristine,
    test_embed_corner_anchored,
    test_embed_centered_offset,
    test_no_spurious_collapse_in_host,
    test_m_less_than_n_raises,
    test_motif_dirname_rules,
    test_compact_representative_preserves_canonical_class,
    test_compact_representative_unwraps_boundary_motif,
    test_centered_embedding_keeps_cbcn3_connected_in_host,
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
