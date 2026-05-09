#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/.benchmarks/obliq/data}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
COLLECTION="${COLLECTION:-obliq_math}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi

"$ROOT/scripts/download_math_dataset.py" --data-dir "$DATA_DIR"
"$ROOT/scripts/setup_math_qdrant.py" \
  --data-dir "$DATA_DIR" \
  --qdrant-url "$QDRANT_URL" \
  --collection "$COLLECTION" \
  "$@"
