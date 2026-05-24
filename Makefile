PYTHON ?= python3
N ?= 5
K ?= 2
OUT ?= generated

.PHONY: help test validate check-dups check-all list-motifs generate-geometry run-legacy render clean

help:
	@echo "Generation:"
	@echo "  list-motifs        — print motif directory names for (N, K)  [no files]"
	@echo "                       example: make list-motifs N=4 K=2"
	@echo "  generate-geometry  — write geometry.in for every distinct motif of (N, K)"
	@echo "                       under \$$OUT/  (default: ./generated/)"
	@echo "                       example: make generate-geometry N=4 K=2 OUT=runs/"
	@echo "  run-legacy         — old defect generator (run.py), pre-canonicalization"
	@echo "  render             — ASCII-render an existing geometry.in"
	@echo "                       example: make render FILE=generated/BN-3-3-1-pure/geometry.in"
	@echo ""
	@echo "Validation:"
	@echo "  test         — algebraic & orbit tests for canonicalize.py (no DFT data needed)"
	@echo "  validate     — verify all maciej_BN/ configs canonicalize into our enumerated classes"
	@echo "  check-dups   — DOS-level comparison of _2_0!C_B vs _2_90!C_B duplicate pairs"
	@echo "  check-all    — run all three above"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean        — remove __pycache__/"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON       — interpreter (default: python3)"
	@echo "  N, K         — supercell size and total defect count (defaults: N=5 K=2)"
	@echo "  OUT          — output directory for generate-geometry (default: generated)"
	@echo "  MACIEJ_DIR   — path to maciej_BN/ (auto-discovered otherwise)"

test:
	$(PYTHON) tests/test_canonicalize.py
	$(PYTHON) tests/test_render.py

validate:
	$(PYTHON) tests/validate_against_maciej.py

check-dups:
	$(PYTHON) tests/check_duplicate_pairs.py

check-all: test validate check-dups

list-motifs:
	$(PYTHON) list_motifs.py $(N) $(K)

generate-geometry:
	$(PYTHON) generate_geometry.py $(N) $(K) $(OUT)

run-legacy:
	$(PYTHON) run.py

render:
	@if [ -z "$(FILE)" ]; then echo "usage: make render FILE=path/to/geometry.in"; exit 2; fi
	$(PYTHON) tools/render_supercell.py $(FILE)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
