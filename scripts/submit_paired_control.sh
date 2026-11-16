#!/usr/bin/env bash
# Submit both paired feature-inversion controls from the existing SAN checkout.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CONDA_PREFIX:?Activate the SAN feature-audit environment first}"
test -x "$CONDA_PREFIX/bin/python"
command -v qsub >/dev/null
task_runtime=/SAN/intelsys/imagenet_mmd_flow/compact_feature_runtime
export TORCH_HOME="$task_runtime/torch"
export XDG_CACHE_HOME="$task_runtime/cache"
export HF_HOME="$task_runtime/hf"
export TMPDIR="$task_runtime/tmp"
mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$TMPDIR"
export FEATURE_AUDIT_SEED=42
export FEATURE_AUDIT_CORRECTION_STEPS=10
export FEATURE_AUDIT_CORRECTION_OBJECTIVE=paired_feature
export FEATURE_AUDIT_STEP_RMS=0.01
task_root=/SAN/intelsys/imagenet_mmd_flow/compact_feature_audits
mkdir -p "$task_root"
task_stamp=$(date +%Y%m%d_%H%M%S)
for task_encoder in mae inception; do
  export FEATURE_AUDIT_ENCODER="$task_encoder"
  export FEATURE_AUDIT_OUTPUT="$task_root/paired_${task_encoder}_seed42_${task_stamp}"
  if [ -e "$FEATURE_AUDIT_OUTPUT" ] || [ -e "${FEATURE_AUDIT_OUTPUT}.log" ]; then
    echo "Output already exists: $FEATURE_AUDIT_OUTPUT" >&2
    exit 1
  fi
  qsub \
    -l 'gpu=true,gpu_type=h100|a100*|v100|a6000|rtx6000ada,tmem=80G,h_rt=7200' \
    -o "${FEATURE_AUDIT_OUTPUT}.log" \
    scripts/audit_sge_1gpu.sh
  echo "Log: ${FEATURE_AUDIT_OUTPUT}.log"
done
