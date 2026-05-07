#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE="${QDRANT_STORAGE:-$ROOT/.benchmarks/qdrant}"
IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:latest}"

mkdir -p "$STORAGE"

if command -v qdrant >/dev/null 2>&1; then
  exec qdrant --storage-dir "$STORAGE"
fi

exec docker run --rm \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$STORAGE:/qdrant/storage" \
  "$IMAGE"
