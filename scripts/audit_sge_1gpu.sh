#!/usr/bin/env bash
#$ -N encoder_audit
#$ -P aihub_ucl
#$ -cwd
#$ -V
#$ -l gpu=true,gpu_type="h100|a100|v100"
#$ -pe gpu 1
#$ -l tmem=80G
#$ -l h_rt=02:00:00
#$ -R y
#$ -j y
#$ -o .
set -euo pipefail
: "${FEATURE_AUDIT_OUTPUT:?Set FEATURE_AUDIT_OUTPUT to a new absolute output directory}"
export DRIFT_COMPILE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
: "${CONDA_PREFIX:?Activate the feature-audit environment before qsub}"
echo "Node: ${HOSTNAME:-unknown}; encoder: ${FEATURE_AUDIT_ENCODER:-mae}; Python: $CONDA_PREFIX/bin/python"
nvidia-smi --query-gpu=name,memory.total --format=csv
exec "$CONDA_PREFIX/bin/python" -u audit_encoders.py \
  --image-root /SAN/intelsys/imagenet_mmd_flow/ILSVRC/Data/CLS-LOC/train \
  --vae /SAN/intelsys/imagenet_mmd_flow/compact_feature_assets/sdvae_hf_root \
  --teacher /SAN/intelsys/imagenet_mmd_flow/compact_feature_assets/models/mae/jax/mae_latent_640 \
  --source-repo "$HOME/riesz_flow_sophia" \
  --encoder "${FEATURE_AUDIT_ENCODER:-mae}" --output "$FEATURE_AUDIT_OUTPUT" \
  --seed "${FEATURE_AUDIT_SEED:-42}" --queries 4 --references 8 --microbatch 1 --repeats 3
