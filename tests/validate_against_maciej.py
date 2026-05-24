"""Validate enumerate_motifs against the maciej_BN/ dataset.

For each directory BN-{n}-{n}-1-{defect}, parse geometry.in, locate the C
substitutions, build a placement tensor, canonicalize it, and check that
the canonical form belongs to our enumeration.
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canonicalize import build_group, canonical, enumerate_motifs
from paths import find_maciej

MACIEJ_DIR = find_maciej()


def parse_geometry(geom_path: Path) -> tuple[int, np.ndarray]:
    """Return (n, tensor) where tensor is shape (2n²,) int8 with 1 at C-substituted sites.

    Atoms in the file alternate B-site, N-site (in their own primitive basis).
    C replaces one of them.

    Maciej's a₂ = (-1/2, √3/2) differs from ours but the lattice is the same;
    canonicalization absorbs the basis choice.
    """
    atoms = []      # list of (frac_x, frac_y, species)
    n = None
    with open(geom_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "lattice_vector" and n is None:
                # |a₁| / |primitive a₁| = n, but we infer n by atom count below
                pass
            if parts[0] == "atom_frac":
                fx, fy, _fz, sp = float(parts[1]), float(parts[2]), float(parts[3]), parts[4]
                atoms.append((fx, fy, sp))
    # Total atoms = 2n²
    n = int(round(np.sqrt(len(atoms) / 2)))
    assert 2 * n * n == len(atoms), f"atom count {len(atoms)} not 2n² for n={n}"

    # Pairs: first atom of each pair is B-sublattice, second is N-sublattice
    # (with C possibly substituting either).
    # Build map: (sublattice, i, j) ← this atom's fractional position
    tensor = np.zeros(2 * n * n, dtype=np.int8)
    for idx, (fx, fy, sp) in enumerate(atoms):
        sublat = idx % 2  # 0 = B, 1 = N
        # Within Maciej's basis, B offset ~ (1/3, 2/3), N offset ~ (2/3, 1/3)
        if sublat == 0:
            off = (1 / 3, 2 / 3)
        else:
            off = (2 / 3, 1 / 3)
        # supercell coords: (fx*n, fy*n) = (i + off_x, j + off_y)
        ci = round(fx * n - off[0])
        cj = round(fy * n - off[1])
        ci %= n
        cj %= n
        if sp == "C":
            tensor[sublat * n * n + ci * n + cj] = 1
    return n, tensor


def parse_dir_name(name: str) -> tuple[int, int, str]:
    m = re.match(r"BN-(\d+)-(\d+)-(\d+)-(.*)", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(4)


def classify(defect: str) -> tuple[int, int] | None:
    """Return (k_B, k_N) for the defect string, or None if ring/pure/unsupported."""
    if defect == "pure":
        return (0, 0)
    if defect in ("benzene", "napthalene", "naphthalene"):
        return None  # ring — handle separately
    if "!" in defect:
        # e.g. C_B_2_0!C_N or C_B_2_90!C_B
        parts = defect.split("!")
        if len(parts) != 2:
            return None  # 3+ defect chains not currently classified
        d1, d2 = parts
        # d1 ends with _{dist}_{angle}
        d1_type = "_".join(d1.split("_")[:2])  # 'C_B' or 'C_N'
        d2_type = d2  # already 'C_B' or 'C_N'
        k_B = (d1_type == "C_B") + (d2_type == "C_B")
        k_N = (d1_type == "C_N") + (d2_type == "C_N")
        return (k_B, k_N)
    if defect == "C_B":
        return (1, 0)
    if defect == "C_N":
        return (0, 1)
    return None


def main():
    dirs = sorted(d for d in MACIEJ_DIR.iterdir() if d.is_dir() and d.name.startswith("BN-"))
    perms_cache: dict[int, np.ndarray] = {}
    enum_cache: dict[tuple[int, int, int], dict] = {}

    n_pass = n_fail = n_skip = 0
    failures = []

    for d in dirs:
        parsed = parse_dir_name(d.name)
        if parsed is None:
            continue
        n, _, defect = parsed
        kbn = classify(defect)
        if kbn is None or "V_" in defect:
            n_skip += 1
            print(f"SKIP  {d.name} (ring/unsupported)")
            continue
        k_B, k_N = kbn

        geom = d / "geometry.in"
        if not geom.exists():
            print(f"SKIP  {d.name} (no geometry.in)")
            n_skip += 1
            continue

        n_geo, tensor = parse_geometry(geom)
        if n_geo != n:
            print(f"FAIL  {d.name} (n mismatch: name={n} geom={n_geo})")
            n_fail += 1
            continue

        if perms_cache.get(n) is None:
            perms_cache[n] = build_group(n)
        perms = perms_cache[n]
        c_actual = canonical(tensor, perms)

        # Check k_B/k_N match
        actual_kB = int(tensor[: n * n].sum())
        actual_kN = int(tensor[n * n :].sum())
        if (actual_kB, actual_kN) != (k_B, k_N):
            print(f"FAIL  {d.name}: name says ({k_B},{k_N}) but geometry has ({actual_kB},{actual_kN})")
            n_fail += 1
            continue

        # Compare against enumeration
        key = (n, k_B, k_N)
        if key not in enum_cache:
            enum_cache[key] = enumerate_motifs(n, k_B, k_N, perms)
        classes = enum_cache[key]

        if c_actual in classes:
            n_pass += 1
            print(f"  ok  {d.name}  (1 of {len(classes)} classes for ({k_B},{k_N}))")
        else:
            n_fail += 1
            print(f"FAIL  {d.name}: canonical form not in enumeration ({len(classes)} classes)")
            failures.append(d.name)

    print(f"\n{n_pass} pass, {n_fail} fail, {n_skip} skip")
    if failures:
        print("FAILURES:", failures)


if __name__ == "__main__":
    main()
