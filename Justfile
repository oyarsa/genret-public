_default:
    @just --list

# Sync dependencies with CPU-only PyTorch
sync-cpu:
    uv sync --extra cpu --extra training

# Sync dependencies with CUDA PyTorch (x86-64 Linux only)
sync-cuda:
    uv sync --extra cuda --extra training

# Run ruff check with autofix
fix:
    uv run ruff check . --fix

# Run ruff check (no fix)
check:
    uv run ruff check .

# Run ruff format
fmt:
    uv run ruff format .

# Run pyrefly (default type checker)
type *files:
    uv run pyrefly check {{ files }}

# Run pyrefly (alias)
pyrefly *files: (type files)

# Run zuban (mypy-compatible type checker)
zuban *files:
    uv run --with zuban zuban check --pretty {{ files }}

# Run pyright
pyright *files:
    uv run --with pyright pyright {{ files }}

# Run ty
ty *files:
    uv run --with ty ty check {{ files }}

# Run all type checkers
type-all *files:
    -just pyright {{ files }}
    @echo
    -just zuban {{ files }}
    @echo
    -just pyrefly {{ files }}
    @echo
    -just ty {{ files }}

# Run all type checkers against baselines
typecheck:
    uv run python scripts/typecheck.py

# Run pre-commit hooks manually on all files
pre-commit:
    uv run pre-commit run --all-files

# Run unit tests
test:
    uv run pytest --quiet tests/

# Run typecheck and tests in parallel
[parallel]
_type-test: test type

# Run ruff check, ruff format, pyrefly and unit tests
lint: fix fmt _type-test

# Check ruff lint, pyrefly and unit tests
check-all: check _type-test

# Watch Python files and run ruff and pyrefly on changes
watch:
    watchexec --exts=py --clear --restart "just check-all"

# Start the graph explorer server (ONNX encoder, torch-free; the dedicated
# entry point skips the full CLI import graph, and --no-sync leaves the
# current venv's extras untouched)
serve:
    uv run --no-sync graft serve

