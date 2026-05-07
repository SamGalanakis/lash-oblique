#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/.benchmarks/obliq/data}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
COLLECTION="${COLLECTION:-obliq_math}"
PYTHON="${OBLIQ_PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv "$ROOT/.venv"
    uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements.txt"
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="${OBLIQ_PYTHON:-python3}"
    "$PYTHON" -m pip install -r "$ROOT/requirements.txt"
  fi
fi

"$PYTHON" "$ROOT/scripts/download_math_dataset.py" --data-dir "$DATA_DIR"
"$PYTHON" "$ROOT/scripts/setup_math_qdrant.py" \
  --data-dir "$DATA_DIR" \
  --qdrant-url "$QDRANT_URL" \
  --collection "$COLLECTION" \
  "$@"
