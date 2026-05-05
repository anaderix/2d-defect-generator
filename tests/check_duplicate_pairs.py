"""Compare DOS between _2_0!C_B and _2_90!C_B pairs.

If our canonicalization is correct, these should be physically equivalent
and yield indistinguishable DOS curves (modulo numerical noise).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import find_maciej

MACIEJ = find_maciej()


def read_dos(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (energy, dos, mu) parsed from KS_DOS_total.dat."""
    mu = None
    with open(path) as f:
        for line in f:
            if "chemical potential" in line.lower() or "mu =" in line:
                # extract number
                for tok in line.split():
                    try:
                        mu = float(tok)
                        break
                    except ValueError:
                        continue
                if mu is not None:
                    break
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 1], mu


def compare(n: int):
    a = MACIEJ / f"BN-{n}-{n}-1-C_B_2_0!C_B" / "KS_DOS_total.dat"
    b = MACIEJ / f"BN-{n}-{n}-1-C_B_2_90!C_B" / "KS_DOS_total.dat"
    if not (a.exists() and b.exists()):
        print(f"n={n}: missing file"); return
    e1, d1, m1 = read_dos(a)
    e2, d2, m2 = read_dos(b)
    # align grids if they differ
    if not np.allclose(e1, e2):
        print(f"n={n}: energy grids differ — interpolating onto common grid")
        emin = max(e1.min(), e2.min()); emax = min(e1.max(), e2.max())
        grid = np.linspace(emin, emax, min(len(e1), len(e2)))
        d1 = np.interp(grid, e1, d1)
        d2 = np.interp(grid, e2, d2)
        e1 = grid
    diff = d1 - d2
    l2 = np.sqrt(np.mean(diff ** 2))
    linf = np.max(np.abs(diff))
    rel_l2 = l2 / np.sqrt(np.mean(d1 ** 2))
    print(
        f"n={n}: μ₁={m1:.4f} μ₂={m2:.4f} Δμ={m2 - m1:+.5f}  "
        f"L∞={linf:.3e}  L²={l2:.3e}  rel L²={rel_l2:.3e}"
    )


if __name__ == "__main__":
    for n in [3, 4, 5, 6, 7]:
        compare(n)
