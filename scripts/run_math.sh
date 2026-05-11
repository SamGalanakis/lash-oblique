#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/.benchmarks/obliq/data}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
COLLECTION="${COLLECTION:-obliq_analogues}"

exec cargo run --release -- run-task \
  --data-dir "$DATA_DIR" \
  --qdrant-url "$QDRANT_URL" \
  --collection "$COLLECTION" \
  "$@"
