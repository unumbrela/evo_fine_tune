#!/usr/bin/env bash
# Stage 4 (LoRA): 使用 LoRA 微调 Evo2 1B
# LoRA 只训练 ~16.8M 参数 (1.5% of 1.1B)，显存需求降低 ~70%
# 适合显存 < 45GB 的 GPU (如 RTX 3090 24GB, RTX 4090 24GB)
#
# 需要 BioNeMo >= 2.7 (LoRA 支持通过 PR #980 合并)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 复用 04_train.sh 的所有配置
CKPT_DIR="${CKPT_DIR:-}"
DATASET_CONFIG="${DATASET_CONFIG:-${PROJECT_DIR}/configs/dataset_blend.yaml}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/data/processed}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_DIR}/results}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-evo2_1b_lora_ft}"
MODEL_SIZE="${MODEL_SIZE:-1b}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
DEVICES="${DEVICES:-1}"
NUM_NODES="${NUM_NODES:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
LR="${LR:-1e-4}"          # LoRA 通常用更小的 LR
MIN_LR="${MIN_LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"

# 自动检测 checkpoint
if [ -z "${CKPT_DIR}" ]; then
    for candidate in \
        "${PROJECT_DIR}/checkpoints/nemo2_evo2_1b_8k" \
        "${PROJECT_DIR}/checkpoints/evo2_1b_8k_bf16" \
        ; do
        if [ -d "${candidate}" ]; then
            CKPT_DIR="${candidate}"
            break
        fi
    done
fi

# 搜索 bionemo cache 目录
if [ -z "${CKPT_DIR}" ]; then
    cache_hit=$(find "${HOME}/.cache/bionemo/" -maxdepth 1 -type d -name "*evo2*1b*untar" 2>/dev/null | head -1)
    if [ -n "${cache_hit}" ]; then
        CKPT_DIR="${cache_hit}"
    fi
fi

if [ -z "${CKPT_DIR}" ]; then
    echo "ERROR: No checkpoint found. Set CKPT_DIR or run scripts/01a_download_checkpoint.sh first."
    exit 1
fi

echo "=== Stage 4 (LoRA): Fine-tuning Evo2 1B ==="
echo "Checkpoint: ${CKPT_DIR}"
echo "LoRA: ~16.8M trainable params (1.5% of total)"
echo "============================================================"

train_evo2 \
    -d "${DATASET_CONFIG}" \
    --dataset-dir "${DATASET_DIR}" \
    --result-dir "${EXPERIMENT_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --model-size "${MODEL_SIZE}" \
    --ckpt-dir "${CKPT_DIR}" \
    --seq-length "${SEQ_LENGTH}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --devices "${DEVICES}" \
    --num-nodes "${NUM_NODES}" \
    --max-steps "${MAX_STEPS}" \
    --lr "${LR}" \
    --min-lr "${MIN_LR}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --lora-finetune \
    --create-tensorboard-logger \
    --val-check-interval 500 \
    --log-every-n-steps 10

echo "=== LoRA training complete ==="
echo "Results: ${EXPERIMENT_DIR}/${EXPERIMENT_NAME}"
