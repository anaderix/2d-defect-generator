"""Locate the maciej_BN/ reference dataset.

Search order:
  1. $MACIEJ_DIR (if set and a directory)
  2. ./maciej_BN/ relative to repo root
  3. ../2d-defects/maciej_BN/ (sibling repo, default)
  4. walk up to 4 levels looking for either of the above
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def find_maciej() -> Path:
    env = os.environ.get("MACIEJ_DIR")
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"MACIEJ_DIR={env!r} is not a directory")

    cur = REPO_ROOT
    for _ in range(4):
        for cand in (cur / "maciej_BN", cur.parent / "2d-defects" / "maciej_BN"):
            if cand.is_dir():
                return cand
        cur = cur.parent

    raise FileNotFoundError(
        "maciej_BN/ not found. Set MACIEJ_DIR env var, or place it next to "
        "this repo (../2d-defects/maciej_BN/)."
    )
