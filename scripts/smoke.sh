#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
task_device=${1:-cpu}
task_root=${2:-runs/synthetic_smoke}
task_cache="$task_root/cache"
python -u cache_teacher.py --synthetic --output "$task_cache" \
  --train-episodes 8 --val-episodes 2 --audit-episodes 2 \
  --queries 4 --references 4 --positions 2 --synthetic-width 4
for task_arm in reconstruction feature geometry; do
  python -u train.py --cache "$task_cache" --config configs/smoke.yaml \
    --arm "$task_arm" --workdir "$task_root/$task_arm" --device "$task_device"
  python -u evaluate.py --cache "$task_cache" --checkpoint "$task_root/$task_arm/latest.pt" \
    --split val --device "$task_device" --output "$task_root/$task_arm/geometry_val.json"
done
echo "Synthetic wiring check complete. No ImageNet quality conclusion can be drawn."
