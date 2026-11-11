#!/usr/bin/env bash
# One-click: extract features -> probe -> optional finetune (RUN_FINETUNE=1).
# Env: DATA_DIR (images), MANIFEST (csv), OUT_DIR (models/, reports/), CUDA_VISIBLE_DEVICES (default 0)
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python3 src/extract_features.py
python3 src/extract_features.py --flip
python3 src/train_probe.py
if [ "${RUN_FINETUNE:-0}" = "1" ]; then python3 src/finetune.py "${FT_BUDGET_S:-1320}"; fi
