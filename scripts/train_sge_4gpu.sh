#!/usr/bin/env bash
#$ -N compact_features_4g
#$ -P aihub_ucl
#$ -cwd
#$ -V
#$ -l gpu=true,gpu_type=h100
#$ -pe gpu 4
#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -R y
#$ -j y
#$ -o .
set -euo pipefail
# The queue/resource names follow Sophia's existing UCL DMMD launcher.
# Run from the cloned repository with the desired conda environment activated.
: "${FEATURE_CACHE:?Set FEATURE_CACHE to the completed teacher episode cache}"
: "${FEATURE_RUN_ROOT:?Set FEATURE_RUN_ROOT to your writable checkpoint/log root}"
task_arm=${FEATURE_ARM:-geometry}
task_config=${FEATURE_CONFIG:-configs/pilot.yaml}
exec python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
  --cache "$FEATURE_CACHE" --config "$task_config" --arm "$task_arm" \
  --workdir "$FEATURE_RUN_ROOT/$task_arm/seed42" --device cuda
