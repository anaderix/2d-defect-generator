# Motif canonicalization for hBN defect configurations

Identifies symmetry-equivalent placements of point defects (and ring motifs)
on a monolayer hBN n×n supercell. Two placements are considered the same
*configuration* iff one can be obtained from the other by:

- periodic translation along the lattice vectors,
- 3-fold rotation `C₃` around a B-site,
- one of three mirrors `σᵥ` through B–N bonds.

Together these form the space group `p3m1` ⊕ translations on n×n with PBC,
of order `|G| = 6n²`. Note that B and N sublattices are **not** exchanged
(no `C₆`, no inversion) — `C_B` and `C_N` are physically distinct.

## Files

```
canonicalize.py             core — symmetry group, lex-min canonicalization,
                            enumeration, ring generators, label maker
test_canonicalize.py        15 algebraic & orbit tests
validate_against_maciej.py  parses maciej_BN/ geometry.in files and verifies
                            each falls into our enumerated set of classes
check_duplicate_pairs.py    DOS-level confirmation that _2_0/_2_90 pairs are
                            physically the same configuration
```

## Tensor representation

A placement is encoded as an `int8` array of length `2·n²`:

```
index 0 …  n²-1     → C_B occupancy on B(i, j),    flat index = i·n + j
index n² … 2n²-1    → C_N occupancy on N(i, j),    flat index = n² + i·n + j
```

Values are `0` (pristine) or `1` (carbon substitution). Ring motifs are
represented as multi-site placements on this same tensor.

## Lattice convention

Primitive vectors `a₁ = (1, 0)`, `a₂ = (1/2, √3/2)`. N atom in cell `(i, j)`
sits at fractional offset `(1/3, 1/3)` relative to B(i, j). Maciej's
`geometry.in` files use a different basis (`a₂ = (-1/2, √3/2)`); the parser
in `validate_against_maciej.py` reconciles this.

## API

### `build_group(n) -> np.ndarray, shape (6n², 2n²)`

Returns an array of permutations indexing the flattened tensor.
Convention: `new_tensor = old_tensor[perm]` — i.e. `perm[m]` is the source
index whose value lands at position `m` after applying the operation.

Group ordering: outer loop over the 6 point operations
`{e, C₃, C₃², σ, σC₃, σC₃²}`, inner loops over the n² translations
`(t_i, t_j) ∈ [0, n)²`.

### `canonical(tensor_flat, perms) -> bytes`

Lex-min over the orbit. Returns the smallest `tensor.tobytes()` string over
all `6n²` group images. Two placements are equivalent iff their canonical
forms are equal.

```python
from canonicalize import build_group, canonical
G = build_group(5)
c1 = canonical(t1, G)
c2 = canonical(t2, G)
equivalent = (c1 == c2)
```

### `enumerate_motifs(n, k_B, k_N, perms=None) -> dict[bytes, np.ndarray]`

All unique canonical configurations with `k_B` C_B substitutions and `k_N`
C_N substitutions on the n×n supercell. Returns a dict keyed by canonical
bytes, with one representative tensor per class.

```python
classes = enumerate_motifs(4, k_B=1, k_N=1)
print(len(classes))   # → 5
```

### `count_classes(n, k) -> dict[(k_B, k_N), int]`

Count of classes for each split of `k = k_B + k_N`.

| n | (0,2) | (1,1) | (2,0) | total |
|---|-------|-------|-------|-------|
| 3 |   2   |   3   |   2   |   7   |
| 4 |   3   |   5   |   3   |  11   |
| 5 |   4   |   7   |   4   |  15   |
| 6 |   6   |   9   |   6   |  21   |
| 7 |   7   |  12   |   7   |  26   |

### `make_benzene(n, i=0, j=0) -> np.ndarray`

Tensor for a benzene ring (6 C atoms = 3 C_B + 3 C_N) centred between cells
`(i, j)` and `(i+1, j+1)`. Pattern:

```
B sites:  (i, j),   (i+1, j),   (i+1, j+1)
N sites:  (i, j),   (i,   j+1), (i+1, j+1)
```

All anchor positions canonicalize to the same class.

### `make_naphthalene(n, i=0, j=0) -> np.ndarray`

Two fused benzene rings, 10 C atoms = 5 C_B + 5 C_N. Pattern derived from
`maciej_BN/BN-5-5-1-napthalene`:

```
B sites:  (i, j),   (i+1, j),   (i+1, j+1),  (i+2, j),    (i+2, j+1)
N sites:  (i, j),   (i,   j+1), (i+1, j),    (i+1, j+1),  (i+2, j+1)
```

### `summary_label(tensor, n, perms=None) -> str`

Human-readable label, compatible with the existing dataset where possible:

```
'pure'                 — no defects
'C_B' / 'C_N'          — single substitution (1 class each, all positions equivalent)
'benzene'              — exact match to make_benzene canonical form
'napthalene'           — exact match to make_naphthalene canonical form
'cb{kB}cn{kN}_{hash6}' — fallback: counts + 6-hex SHA suffix of canonical bytes
```

The hash is stable: equivalent placements always produce the same suffix.

### `motif_dirname(tensor, n, perms=None) -> str`

Full directory name `BN-{n}-{n}-1-{summary_label}`.

### Helpers

- `build_coords(n) -> ndarray, shape (2n², 2)` — Cartesian xy of all sites
- `nearest_neighbours(n) -> set[frozenset]` — B–N bonds (with PBC)

## Validation

```bash
python test_canonicalize.py           # 15/15 tests
python validate_against_maciej.py     # 35/35 non-ring + 10 rings (skipped/handled)
python check_duplicate_pairs.py       # DOS comparison of _2_0 vs _2_90 pairs
```

Findings:

1. All 45 maciej_BN configurations canonicalize into our enumerated classes
   (35 point-defect + 10 ring motifs).
2. The `_2_0!C_B` and `_2_90!C_B` directories are duplicates: both fall into
   the same canonical class for every `n ∈ {3..7}`, and DOS curves agree to
   ≤ 0.5% relative L² with chemical potentials matching to ≤ 10⁻⁴ eV. The
   "90" in the filename actually corresponds to a 120° angle (related to
   "0" by `C₃`, which is a symmetry of hBN).
3. Maciej's dataset covers a small subset of the orbit space: e.g. for
   `n=7, k=2` the dataset has 3 unique classes, our enumeration finds 26.

## Scaling

| operation                                    | complexity                          |
|----------------------------------------------|-------------------------------------|
| canonicalize one placement                   | `O(\|G\|·n²) = O(n⁴)`              |
| enumerate all classes for (n, k_B, k_N)      | `O(C(n², k_B)·C(n², k_N)·n⁴)`       |

Practical limits on a single CPU:

| n  | max k for full enumeration | dominant cost                   |
|----|---------------------------|---------------------------------|
| 7  | ≲ 6                       | C(49, 3)·C(49, 3) = 1.3·10⁸     |
| 16 | ≲ 3                       | C(256, 3)·C(256, 3) ≈ 4·10¹³    |

For larger `(n, k)`, exhaustive enumeration is infeasible — switch to
sampling or constructive orbit generation (Read–Faradžev / Nauty).

## Limitations

- **Single layer only.** Multi-layer stacking (AA, AB, AB′) not handled.
- **No support for vacancies or non-C substituents** in the current API
  (extension is straightforward — add a third value `-1` to the tensor).
- **Ring motifs are anchored patterns.** Other multi-atom motifs (e.g.
  rotated naphthalene, anthracene, branched chains) need explicit
  generators if they are to be detected by `summary_label`.
- **Hash labels are not orderable by physical meaning.** For research
  workflows you may want a separate map: canonical_bytes → (shell index,
  angle index, …) tailored to your downstream use.
