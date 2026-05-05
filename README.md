# 2d-defect-generator

Generates FHI-aims `geometry.in` files for monolayer hBN supercells with
defect configurations. Two pipelines coexist:

- **Canonical pipeline** (`canonicalize.py` + `generate_geometry.py`):
  enumerates every symmetry-distinct C-substitution motif on n×n hBN under
  the space group `p3m1`, writing one `geometry.in` per equivalence class.
  Documented in [`README_motif.md`](README_motif.md).

- **Legacy pipeline** (`run.py` + `defect_gen_class.py`): pre-existing
  generator with hand-named defects (`C_B`, `C_B_2_0!C_N`, `benzene`, …).
  Does not deduplicate symmetry-equivalent placements.

## Quick start

```bash
make help                              # list targets
make test                              # run algebraic tests for canonicalize.py
make generate-geometry N=4 K=2         # write 11 geometry.in files under generated/
make list-motifs N=4 K=2               # just print the motif names, no files
```

## Makefile targets

The Makefile is the single entry point for everything. All targets pass
through `$(PYTHON)` (default `python3`); set `PYTHON=…` to use a specific
interpreter.

### Generation

| target              | does what                                                   |
|---------------------|-------------------------------------------------------------|
| `list-motifs`       | print directory names for every distinct motif at `(N, K)` |
| `generate-geometry` | write `geometry.in` for every distinct motif under `$(OUT)` |
| `run-legacy`        | invoke the old `run.py` defect generator                    |

`list-motifs` calls `list_motifs.py N K`. It enumerates all `(k_B, k_N)`
splits with `k_B + k_N = K` and prints one motif name per line, grouped by
split. No files are written.

`generate-geometry` calls `generate_geometry.py N K $(OUT)`. For each
canonical class, it writes `$(OUT)/BN-{N}-{N}-1-{label}/geometry.in`. The
`{label}` is human-readable (`pure`, `C_B`, `C_N`, `benzene`, `napthalene`)
where a known pattern is detected, otherwise `cb{kB}cn{kN}_{hash6}` with a
6-hex SHA suffix that is stable across runs.

`run-legacy` runs the older generator unchanged. Use it only if you need
specific named defects from its DSL; for systematic enumeration prefer
`generate-geometry`.

### Validation

| target       | does what                                                                  |
|--------------|----------------------------------------------------------------------------|
| `test`       | algebraic & orbit tests for the symmetry group (no DFT data needed)        |
| `validate`   | checks every `maciej_BN/BN-…/geometry.in` falls into our enumerated classes |
| `check-dups` | DOS-level comparison of `_2_0!C_B` vs `_2_90!C_B` duplicate pairs          |
| `check-all`  | runs all three                                                             |

These each invoke a script under `tests/`. The two that touch the
reference dataset (`validate`, `check-dups`) auto-discover `maciej_BN/`
via `paths.find_maciej()`:

1. `$MACIEJ_DIR` if set
2. `./maciej_BN/`
3. `../2d-defects/maciej_BN/` (default sibling repo)
4. walk up to 4 levels

Override with `MACIEJ_DIR=/path/to/maciej_BN make validate`.

### Maintenance

| target  | does what                       |
|---------|---------------------------------|
| `clean` | remove `__pycache__/` directories |

## Variables

| variable     | default     | used by                              |
|--------------|-------------|--------------------------------------|
| `PYTHON`     | `python3`   | every target                         |
| `N`          | `5`         | `list-motifs`, `generate-geometry`   |
| `K`          | `2`         | `list-motifs`, `generate-geometry`   |
| `OUT`        | `generated` | `generate-geometry`                  |
| `MACIEJ_DIR` | (auto)      | `validate`, `check-dups`             |

Override on the command line: `make generate-geometry N=6 K=3 OUT=runs/`.

## Repository layout

```
canonicalize.py         core: symmetry group, lex-min, enumeration, labels
generate_geometry.py    write geometry.in per canonical class
list_motifs.py          print motif names without writing files
paths.py                find_maciej() — locate the reference dataset
defect_gen_class.py     legacy generator
run.py                  legacy entry point
BN.in                   primitive hBN cell (lattice + B/N atoms)
control.in              FHI-aims SCF parameters template (legacy)

Makefile                single entry point
README.md               this file
README_motif.md         deeper docs on the canonicalization logic

tests/
  test_canonicalize.py        15 algebraic & orbit tests
  validate_against_maciej.py  35 round-trip checks against the reference dataset
  check_duplicate_pairs.py    DOS comparison of _2_0 vs _2_90 pairs
```

## Adding a new motif type

If the canonicalization is correct and you need a new named pattern (e.g.
anthracene, vacancies), add a generator function next to `make_benzene()`
in `canonicalize.py`, register it in `_ring_canons` so `summary_label`
recognizes it, and write a small test that confirms the generator's
canonical form is invariant across translations.
