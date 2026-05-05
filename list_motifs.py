"""List all symmetry-distinct motif directory names for a given supercell size
and number of defects.

Usage:
    python list_motifs.py <n> <k>           # all (k_B, k_N) splits with k_B+k_N=k
    python list_motifs.py <n> <k_B> <k_N>   # specific split
"""

import sys

from canonicalize import (
    build_group,
    enumerate_motifs,
    motif_dirname,
)


def list_motifs(n: int, k_B: int, k_N: int) -> list[str]:
    perms = build_group(n)
    classes = enumerate_motifs(n, k_B, k_N, perms)
    return sorted(motif_dirname(t, n, perms) for t in classes.values())


def main(argv: list[str]) -> int:
    if len(argv) == 3:
        n = int(argv[1])
        k = int(argv[2])
        splits = [(k_B, k - k_B) for k_B in range(k + 1)]
    elif len(argv) == 4:
        n = int(argv[1])
        splits = [(int(argv[2]), int(argv[3]))]
    else:
        print(__doc__, file=sys.stderr)
        return 2

    total = 0
    for k_B, k_N in splits:
        names = list_motifs(n, k_B, k_N)
        print(f"# (k_B={k_B}, k_N={k_N}): {len(names)} class(es)")
        for name in names:
            print(name)
        total += len(names)
    print(f"# total: {total}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
