"""ASCII render of an n×n hBN supercell in armchair-vertical orientation.

Lattice convention matches BN.in: a₁ = (1, 0), a₂ = (-1/2, √3/2).

The renderer has three entry points:
  - render_from_grid(B_glyphs, N_glyphs)  — fundamental; takes n×n char arrays
  - render_from_tensor(tensor, n)          — for generate_geometry.py
  - render_from_geometry_file(path)        — for the CLI (uses ASE)

CLI:
  python tools/render_supercell.py path/to/geometry.in [path2 ...]

Glyphs:
  'B', 'N'   pristine atoms
  'C'        carbon substitution
  '.'        vacancy (missing atom)
"""
from __future__ import annotations
import numpy as np

# Layout constants — chosen so 1 primitive cell is 8 cols × 4 rows, which
# happens to render as a near-square visual block in a 2:1 character cell.
CELL_W = 8   # cols per primitive cell along a₁
CELL_H = 4   # rows per primitive cell along a₂ (vertical)
A2_DC = -4   # col shift per +a₂ step (a₂.x = -1/2 → negative)
A2_DR = +4   # row shift per +a₂ step (downward in display)


def render_from_grid(B_glyphs: np.ndarray, N_glyphs: np.ndarray) -> str:
    """Render two n×n grids of one-character glyphs as an ASCII supercell.

    B_glyphs[i, j] is the glyph at B-sublattice cell (i, j); likewise N_glyphs.
    """
    n = B_glyphs.shape[0]
    assert B_glyphs.shape == (n, n) == N_glyphs.shape

    col_shift = -((n - 1) * A2_DC)
    cols = (n - 1) * CELL_W + CELL_W // 2 + col_shift + 2
    rows = (n - 1) * A2_DR + CELL_H // 2 + 2

    grid = [[" "] * cols for _ in range(rows)]

    def place_atom(c, r, glyph):
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = glyph

    def place_bond(c, r, glyph):
        if 0 <= r < rows and 0 <= c < cols and grid[r][c] == " ":
            grid[r][c] = glyph

    # Atom positions: B at (col, row_top); N at (col + W/2, row_top + H/2)
    atom_pos = {}
    for i in range(n):
        for j in range(n):
            row_top = (n - 1 - j) * A2_DR
            col_B = i * CELL_W + j * A2_DC + col_shift
            col_N = col_B + CELL_W // 2
            row_N = row_top + CELL_H // 2
            atom_pos[(0, i, j)] = (col_B, row_top)
            atom_pos[(1, i, j)] = (col_N, row_N)
            place_atom(col_B, row_top, str(B_glyphs[i, j]))
            place_atom(col_N, row_N, str(N_glyphs[i, j]))

    # Bonds — B(i,j) NN N-sites: N(i,j), N(i-1,j), N(i,j+1).
    nn_offsets = [(0, 0), (-1, 0), (0, 1)]

    def short_bond(cB, rB, dc, dr):
        if dr == 2 and dc == 4:    return (cB + 2, rB + 1, "\\")
        if dr == 2 and dc == -4:   return (cB - 2, rB + 1, "/")
        if dr == 2 and dc == 0:    return (cB, rB + 1, "|")
        if dr == -2 and dc == 4:   return (cB + 2, rB - 1, "/")
        if dr == -2 and dc == -4:  return (cB - 2, rB - 1, "\\")
        if dr == -2 and dc == 0:   return (cB, rB - 1, "|")
        return None

    pbc_shifts = [
        (0, 0),
        (n * CELL_W, 0), (-n * CELL_W, 0),
        (n * A2_DC, -n * A2_DR), (-n * A2_DC, n * A2_DR),
    ]

    for i in range(n):
        for j in range(n):
            cB, rB = atom_pos[(0, i, j)]
            for di, dj in nn_offsets:
                ni, nj = (i + di) % n, (j + dj) % n
                cN, rN = atom_pos[(1, ni, nj)]
                dc, dr = cN - cB, rN - rB
                for shift_c, shift_r in pbc_shifts:
                    bond = short_bond(cB, rB, dc + shift_c, dr + shift_r)
                    if bond:
                        place_bond(*bond)
                        break

    lines = ["".join(row).rstrip() for row in grid]
    # Drop trailing all-blank lines so callers don't have to special-case them.
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def render_from_tensor(tensor: np.ndarray, n: int) -> str:
    """Render the (2n²,) tensor used by canonicalize.py / generate_geometry.py.

    Tensor encoding:
      index i*n + j           → 1 if C substitutes B(i, j), else 0 (B remains)
      index n² + i*n + j      → 1 if C substitutes N(i, j), else 0 (N remains)

    Vacancies are not representable in this tensor and are not rendered here.
    """
    m = n * n
    B_g = np.full((n, n), "B", dtype="<U1")
    N_g = np.full((n, n), "N", dtype="<U1")
    for i in range(n):
        for j in range(n):
            if tensor[i * n + j]:
                B_g[i, j] = "C"
            if tensor[m + i * n + j]:
                N_g[i, j] = "C"
    return render_from_grid(B_g, N_g)


def render_from_geometry_file(path: str, n_hint: int | None = None) -> str:
    """Render a geometry.in file. Requires ASE (only used in the CLI)."""
    from ase.io import read
    atoms = read(path, format="aims")
    return _render_atoms(atoms, n_hint=n_hint)


def _render_atoms(atoms, n_hint=None) -> str:
    """Parse an ASE Atoms object into B/N grids (handling vacancies) and render."""
    syms = atoms.get_chemical_symbols()
    fracs = atoms.get_scaled_positions()[:, :2]
    if n_hint is None:
        n_guess = int(round(np.sqrt(len(atoms) / 2)))
        n = n_guess if 2 * n_guess * n_guess >= len(atoms) else n_guess + 1
    else:
        n = n_hint

    B_g = np.full((n, n), "B", dtype="<U1")
    N_g = np.full((n, n), "N", dtype="<U1")
    present_B = np.zeros((n, n), dtype=bool)
    present_N = np.zeros((n, n), dtype=bool)
    for frac, sp in zip(fracs, syms):
        for sublat, off in enumerate([(1/3, 2/3), (2/3, 1/3)]):
            ci = round(frac[0] * n - off[0]) % n
            cj = round(frac[1] * n - off[1]) % n
            exp_x = (ci + off[0]) / n
            exp_y = (cj + off[1]) / n
            if (abs((frac[0] - exp_x + 0.5) % 1.0 - 0.5) < 0.05 and
                abs((frac[1] - exp_y + 0.5) % 1.0 - 0.5) < 0.05):
                glyph = "C" if sp == "C" else ("B" if sublat == 0 else "N")
                if sublat == 0:
                    B_g[ci, cj] = glyph
                    present_B[ci, cj] = True
                else:
                    N_g[ci, cj] = glyph
                    present_N[ci, cj] = True
                break
    B_g[~present_B] = "."
    N_g[~present_N] = "."
    return render_from_grid(B_g, N_g)


def annotate(B_g: np.ndarray | None = None,
             N_g: np.ndarray | None = None,
             tensor: np.ndarray | None = None,
             n: int | None = None) -> str:
    """Short one- or two-line annotation listing C / vacancy positions.

    Accepts either explicit grids or a tensor+n.
    """
    if tensor is not None and n is not None:
        m = n * n
        B_g = np.full((n, n), "B", dtype="<U1")
        N_g = np.full((n, n), "N", dtype="<U1")
        for i in range(n):
            for j in range(n):
                if tensor[i * n + j]:
                    B_g[i, j] = "C"
                if tensor[m + i * n + j]:
                    N_g[i, j] = "C"
    assert B_g is not None and N_g is not None
    n = B_g.shape[0]

    out = []
    for label, grid in [("B", B_g), ("N", N_g)]:
        c_sites = [(i, j) for j in range(n) for i in range(n) if grid[i, j] == "C"]
        v_sites = [(i, j) for j in range(n) for i in range(n) if grid[i, j] == "."]
        if c_sites:
            out.append(f"C on {label}-sites: " +
                       ", ".join(f"({i},{j})" for i, j in c_sites))
        if v_sites:
            out.append(f"V on {label}-sites: " +
                       ", ".join(f"({i},{j})" for i, j in v_sites))
    if not out:
        out.append("(pristine hBN)")
    return "\n".join(out)


def _cli():
    import sys
    if len(sys.argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    for path in sys.argv[1:]:
        label = path.split("/")[-2] if "/" in path else path
        print(f"# === {label} ===")
        for line in render_from_geometry_file(path).split("\n"):
            print(f"# {line}")
        print()


if __name__ == "__main__":
    _cli()
