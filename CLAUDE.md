# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

The `Makefile` is the single entry point. All targets honor `PYTHON` (default `python3`),
`N`/`K` (sub-supercell size and total defect count), `M` (host supercell size, defaults
to `N`), and `OUT` (output dir).

```bash
make help                              # list targets and variables
make test                              # algebraic & orbit tests for canonicalize.py (no DFT data)
make list-motifs N=4 K=2               # print motif directory names, no files written
make generate-geometry N=4 K=2         # write geometry.in for each canonical class under generated/
make validate                          # round-trip check against maciej_BN/ reference dataset
make check-dups                        # DOS comparison of _2_0!C_B vs _2_90!C_B duplicate pairs
make check-all                         # test + validate + check-dups
make run-legacy                        # invoke old run.py defect generator
make render FILE=path/to/geometry.in   # ASCII-render any existing geometry.in
```

Run a single canonicalization test directly:
```bash
python3 tests/test_canonicalize.py
```

`validate` and `check-dups` need the `maciej_BN/` reference dataset. They auto-discover via
`paths.find_maciej()`: `$MACIEJ_DIR` → `./maciej_BN/` → `../2d-defects/maciej_BN/` → walk up 4
levels. Override with `MACIEJ_DIR=/path/to/maciej_BN make validate`.

## Architecture

Two coexisting pipelines for generating FHI-aims `geometry.in` files for monolayer hBN
supercells with C-substitution defects:

### Canonical pipeline (preferred for systematic enumeration)

`canonicalize.py` → `generate_geometry.py` / `list_motifs.py`

The core abstraction is a flat `int8` tensor of length `2n²` representing carbon
substitution occupancy on an n×n hBN supercell:
- index `i*n + j`        → C_B occupancy on B-sublattice cell `(i, j)`
- index `n² + i*n + j`    → C_N occupancy on N-sublattice cell `(i, j)`

Two placements are *equivalent* iff related by an element of the space group
**`p3m1` ⊕ translations**, order `|G| = 6n²`: 3-fold rotation around a B-site, three mirror
planes through B–N bonds, plus n² periodic translations. **B and N sublattices are not
exchanged** (no C₆, no inversion) — `C_B` and `C_N` are physically distinct.

`build_group(n)` materializes all `6n²` permutations as a `(6n², 2n²)` array with the
convention `new_tensor = old_tensor[perm]`. `canonical(t, perms)` returns the lex-min over
the orbit as `bytes`; equivalence is byte equality. `enumerate_motifs(n, k_B, k_N)` walks
combinations and dedupes by canonical key.

`summary_label()` produces human-readable names compatible with the existing dataset where
possible: `'pure'`, `'C_B'`, `'C_N'`, `'benzene'`, `'napthalene'`, falling back to
`'cb{kB}cn{kN}_{hash6}'` (6-hex SHA of canonical bytes — stable across runs). Adding a new
named motif means adding a generator like `make_benzene()` and registering its canonical
form in `_ring_canons`.

### Legacy pipeline

`run.py` → `defect_gen_class.py`. Hand-named defects (`C_B`, `C_B_2_0!C_N`, `benzene`, …),
no symmetry deduplication. Uses `BN.in` (primitive cell) and `control.in` (FHI-aims SCF
template). Use only when you need its specific named DSL; otherwise prefer the canonical
pipeline.

### Lattice convention

`a₁ = (1, 0)`, `a₂ = (1/2, √3/2)`. N atom in cell `(i, j)` sits at fractional offset
`(1/3, 1/3)` relative to B(i, j). The reference `maciej_BN/` files use a different basis
(`a₂ = (-1/2, √3/2)`); `tests/validate_against_maciej.py` reconciles this.

### Output naming

`generate_geometry.py` writes one `geometry.in` per canonical class. The `1` in the
directory name is the layer count (single layer only — multi-layer stacking is not
supported in the canonical pipeline). Directory naming depends on the host-cell size `M`:

- `BN-{m}-{m}-1-pure/` — pristine host (any `k = 0`, regardless of `n`).
- `BN-{n}-{n}-1-{label}/` — when `m == n` (no embedding).
- `BN-{m}-{m}-1-N{n}-{label}/` — when `m > n` (motif embedded at corner `(0,0)`,
  remaining cells pristine; hash label stays computed on the `n × n` tensor so it's
  stable across `m`).

Each emitted `geometry.in` carries an ASCII picture of the (full m×m) supercell in its
header (armchair-vertical orientation), generated via `tools/render_supercell.py`. The
renderer also has a standalone CLI for inspecting any existing `geometry.in` (including
legacy `maciej_BN/` files with vacancies, shown as `.`).

## Scaling notes

- Canonicalize one placement: `O(|G|·n²) = O(n⁴)`.
- Enumerate `(n, k_B, k_N)`: `O(C(n², k_B)·C(n², k_N)·n⁴)`.
- Practical limit on a single CPU: roughly `n=7, k≲6` or `n=16, k≲3`. For larger cases,
  exhaustive enumeration is infeasible; switch to sampling or constructive orbit
  generation (Read–Faradžev / Nauty).

## Deeper docs

`README_motif.md` covers the canonicalization API, tensor encoding, ring-motif anchors,
known limitations (single layer only, no vacancies, anchored ring patterns), and the
class-count table for `n ∈ {3..7}`.
