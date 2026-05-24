"""Snapshot tests for tools.render_supercell.

These lock in the ASCII output for a few known configurations so that
accidental regressions in glyph placement, slant direction, or bond drawing
get caught. To intentionally update a snapshot, edit the expected string
below to match the new output.

No ASE dependency — uses render_from_tensor / render_from_grid directly.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.render_supercell import (
    annotate,
    render_from_grid,
    render_from_tensor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_eq(name: str, got: str, expected: str) -> None:
    if got == expected:
        print(f"ok  {name}")
        return
    print(f"FAIL {name}")
    g = got.splitlines()
    e = expected.splitlines()
    for i in range(max(len(g), len(e))):
        gi = g[i] if i < len(g) else "<missing>"
        ei = e[i] if i < len(e) else "<missing>"
        marker = "  " if gi == ei else " *"
        print(f"  {marker} {i:2d} got     |{gi}|")
        print(f"  {marker}    expected|{ei}|")
    raise AssertionError(name)


def _tensor(n: int, B_C=(), N_C=()) -> np.ndarray:
    """Build a (2n²,) tensor with C substitutions at the listed (i,j) sites."""
    t = np.zeros(2 * n * n, dtype=np.int8)
    for i, j in B_C:
        t[i * n + j] = 1
    for i, j in N_C:
        t[n * n + i * n + j] = 1
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pure_n3():
    """3×3 pristine hBN — all atoms B and N, no substitutions."""
    got = render_from_tensor(_tensor(3), n=3)
    expected = (
        "B       B       B\n"
        "  \\   /   \\   /   \\\n"
        "    N       N       N\n"
        "    |       |       |\n"
        "    B       B       B\n"
        "  /   \\   /   \\   /   \\\n"
        "        N       N       N\n"
        "        |       |       |\n"
        "        B       B       B\n"
        "      /   \\   /   \\   /   \\\n"
        "            N       N       N"
    )
    _assert_eq("test_pure_n3", got, expected)


def test_single_C_B_n3():
    """3×3 with one C on B-site at (0,0). Locks in vertex position & glyph."""
    got = render_from_tensor(_tensor(3, B_C=[(0, 0)]), n=3)
    # C(0,0) should be at the bottom-left of the slanted parallelogram.
    expected = (
        "B       B       B\n"
        "  \\   /   \\   /   \\\n"
        "    N       N       N\n"
        "    |       |       |\n"
        "    B       B       B\n"
        "  /   \\   /   \\   /   \\\n"
        "        N       N       N\n"
        "        |       |       |\n"
        "        C       B       B\n"
        "      /   \\   /   \\   /   \\\n"
        "            N       N       N"
    )
    _assert_eq("test_single_C_B_n3", got, expected)


def test_NN_dimer_n3():
    """C_B at (0,0) + C_N at (0,0) → adjacent C–C pair sharing the δ_BN bond."""
    got = render_from_tensor(_tensor(3, B_C=[(0, 0)], N_C=[(0, 0)]), n=3)
    expected = (
        "B       B       B\n"
        "  \\   /   \\   /   \\\n"
        "    N       N       N\n"
        "    |       |       |\n"
        "    B       B       B\n"
        "  /   \\   /   \\   /   \\\n"
        "        N       N       N\n"
        "        |       |       |\n"
        "        C       B       B\n"
        "      /   \\   /   \\   /   \\\n"
        "            C       N       N"
    )
    _assert_eq("test_NN_dimer_n3", got, expected)


def test_vacancy_via_grid():
    """Render_from_grid directly so we can exercise the vacancy glyph '.'.
    Place a B-vacancy at (0,0), keep everything else pristine."""
    n = 3
    B_g = np.full((n, n), "B", dtype="<U1")
    N_g = np.full((n, n), "N", dtype="<U1")
    B_g[0, 0] = "."
    got = render_from_grid(B_g, N_g)
    # The '.' should appear exactly where C_B would be in test_single_C_B_n3.
    assert ".       B       B" in got, f"vacancy glyph not at expected location:\n{got}"
    assert "C" not in got, "no C should appear in this config"
    print("ok  test_vacancy_via_grid")


def test_annotate_lists_C_and_V():
    n = 3
    B_g = np.full((n, n), "B", dtype="<U1")
    N_g = np.full((n, n), "N", dtype="<U1")
    B_g[0, 0] = "C"
    B_g[1, 2] = "."
    N_g[2, 1] = "C"
    text = annotate(B_g=B_g, N_g=N_g)
    assert "C on B-sites: (0,0)" in text
    assert "V on B-sites: (1,2)" in text
    assert "C on N-sites: (2,1)" in text
    print("ok  test_annotate_lists_C_and_V")


def test_annotate_pristine():
    n = 3
    text = annotate(tensor=_tensor(n), n=n)
    assert text == "(pristine hBN)"
    print("ok  test_annotate_pristine")


def test_grid_dimensions_scale_with_n():
    """Sanity check that wider supercells produce wider/taller renders."""
    r3 = render_from_tensor(_tensor(3), n=3).splitlines()
    r5 = render_from_tensor(_tensor(5), n=5).splitlines()
    assert len(r5) > len(r3), "n=5 should have more rows than n=3"
    max_w3 = max(len(line) for line in r3)
    max_w5 = max(len(line) for line in r5)
    assert max_w5 > max_w3, "n=5 should be wider than n=3"
    print(f"ok  test_grid_dimensions_scale_with_n  (n=3: {len(r3)}x{max_w3}, n=5: {len(r5)}x{max_w5})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_pure_n3,
    test_single_C_B_n3,
    test_NN_dimer_n3,
    test_vacancy_via_grid,
    test_annotate_lists_C_and_V,
    test_annotate_pristine,
    test_grid_dimensions_scale_with_n,
]


def main() -> int:
    passed = failed = 0
    for t in TESTS:
        try:
            t()
            passed += 1
        except AssertionError:
            failed += 1
    print(f"\n{passed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
