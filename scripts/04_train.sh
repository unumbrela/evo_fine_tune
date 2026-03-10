#!/usr/bin/env bash
# Stage 4: 微调 Evo2 1B
# 需要在 BioNeMo 容器内运行，且有 GPU
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================================================
# 配置参数 (通过环境变量覆盖)
# ============================================================

# -- Checkpoint --
# BF16 版本 (A100/A6000 可用):
#   使用 01a 下载的路径, 或 bionemo.core.data.load 返回的路径
# FP8 版本 (仅 H100):
#   改用 evo2/1b-8k:1.0 对应的路径
CKPT_DIR="${CKPT_DIR:-}"

# 如果未指定 checkpoint 路径，尝试自动检测
if [ -z "${CKPT_DIR}" ]; then
    # 尝试常见路径
    for candidate in \
        "${PROJECT_DIR}/checkpoints/nemo2_evo2_1b_8k" \
        "${PROJECT_DIR}/checkpoints/evo2_1b_8k_bf16" \
        "${HOME}/.cache/bionemo/evo2_1b-8k-bf16_1.0" \
        ; do
        if [ -d "${candidate}" ]; then
            CKPT_DIR="${candidate}"
            break
        fi
    done
fi

if [ -z "${CKPT_DIR}" ]; then
    echo "ERROR: No checkpoint found. Set CKPT_DIR or run scripts/01a_download_checkpoint.sh first."
    exit 1
fi

# -- 数据 --
DATASET_CONFIG="${DATASET_CONFIG:-${PROJECT_DIR}/configs/dataset_blend.yaml}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/data/processed}"

# -- 输出 --
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_DIR}/results}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-evo2_1b_ft}"

# -- 模型 --
MODEL_SIZE="${MODEL_SIZE:-1b}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"

# -- 训练超参 --
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"    # OOM 时减小此值
DEVICES="${DEVICES:-1}"                       # GPU 数量
NUM_NODES="${NUM_NODES:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
WD="${WD:-0.01}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"

# -- 验证/日志 --
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-20}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"

# -- 精度 --
# 如果 GPU 支持 FP8 (H100), 添加 --fp8 可加速训练
USE_FP8="${USE_FP8:-false}"

# -- 日志 --
USE_WANDB="${USE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-evo2-finetune}"

# ============================================================
echo "=== Stage 4: Fine-tuning Evo2 1B ==="
echo "Checkpoint: ${CKPT_DIR}"
echo "Dataset config: ${DATASET_CONFIG}"
echo "Model size: ${MODEL_SIZE}"
echo "Devices: ${DEVICES} x ${NUM_NODES} nodes"
echo "Max steps: ${MAX_STEPS}"
echo "Micro batch size: ${MICRO_BATCH_SIZE}"
echo "Learning rate: ${LR}"
echo "FP8: ${USE_FP8}"
echo "============================================================"

# 构建命令
CMD="train_evo2"
CMD+=" -d ${DATASET_CONFIG}"
CMD+=" --dataset-dir ${DATASET_DIR}"
CMD+=" --experiment-dir ${EXPERIMENT_DIR}"
CMD+=" --experiment-name ${EXPERIMENT_NAME}"
CMD+=" --model-size ${MODEL_SIZE}"
CMD+=" --ckpt-dir ${CKPT_DIR}"
CMD+=" --seq-length ${SEQ_LENGTH}"
CMD+=" --micro-batch-size ${MICRO_BATCH_SIZE}"
CMD+=" --devices ${DEVICES}"
CMD+=" --num-nodes ${NUM_NODES}"
CMD+=" --max-steps ${MAX_STEPS}"
CMD+=" --lr ${LR}"
CMD+=" --min-lr ${MIN_LR}"
CMD+=" --warmup-steps ${WARMUP_STEPS}"
CMD+=" --wd ${WD}"
CMD+=" --clip-grad ${CLIP_GRAD}"
CMD+=" --val-check-interval ${VAL_CHECK_INTERVAL}"
CMD+=" --limit-val-batches ${LIMIT_VAL_BATCHES}"
CMD+=" --log-every-n-steps ${LOG_EVERY_N_STEPS}"

# FP8
if [ "${USE_FP8}" = "true" ]; then
    CMD+=" --fp8"
fi

# Wandb
if [ "${USE_WANDB}" = "true" ]; then
    CMD+=" --wandb-project ${WANDB_PROJECT}"
else
    CMD+=" --no-wandb"
fi

# Tensorboard (默认开启)
CMD+=" --create-tensorboard-logger"

echo ""
echo "Running: ${CMD}"
echo ""

eval ${CMD}

echo ""
echo "=== Training complete ==="
echo "Results: ${EXPERIMENT_DIR}/${EXPERIMENT_NAME}"
