#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/train_scafell.sh CACHE ARM WORKDIR" >&2
  exit 2
fi
exec python -u train.py --cache "$1" --arm "$2" --workdir "$3" \
  --config configs/pilot.yaml --device cuda
