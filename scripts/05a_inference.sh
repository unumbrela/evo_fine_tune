#!/usr/bin/env bash
# Stage 5a: 使用微调后的模型进行序列生成
# 需要在 BioNeMo 容器内运行
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Checkpoint 目录
# - 全量微调: 直接使用微调后的 checkpoint
# - LoRA 微调: infer_evo2 不支持 LoRA，需使用基础模型 checkpoint
#   (LoRA 推理请用 predict_evo2 或自定义脚本)
if [ -z "${CKPT_DIR:-}" ]; then
    # 优先使用全量微调 checkpoint
    if [ -d "${PROJECT_DIR}/results/evo2_1b_ft/checkpoints" ]; then
        latest=$(find "${PROJECT_DIR}/results/evo2_1b_ft/checkpoints" -maxdepth 2 -type d -name "context" 2>/dev/null \
            | sed 's|/context$||' | sort | tail -1)
        [ -n "${latest}" ] && CKPT_DIR="${latest}"
    fi
    # 回退到基础模型 checkpoint
    if [ -z "${CKPT_DIR:-}" ]; then
        for candidate in \
            "${PROJECT_DIR}/checkpoints/nemo2_evo2_1b_8k" \
            "${PROJECT_DIR}/checkpoints/evo2_1b_8k_bf16" \
            ; do
            [ -d "${candidate}" ] && CKPT_DIR="${candidate}" && break
        done
    fi
    # 搜索 bionemo cache
    if [ -z "${CKPT_DIR:-}" ]; then
        cache_hit=$(find "${HOME}/.cache/bionemo/" -maxdepth 1 -type d -name "*evo2*1b*untar" 2>/dev/null | head -1)
        [ -n "${cache_hit}" ] && CKPT_DIR="${cache_hit}"
    fi
fi
CKPT_DIR="${CKPT_DIR:?ERROR: No checkpoint found. Set CKPT_DIR or run training first.}"

# 输入 prompt (DNA 序列片段)
PROMPT="${PROMPT:-ATGAAAGCAATTTTCGTACTGAAAATTTCGATTCTTACCTTAGCAGGCG}"

# 生成参数
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-500}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-4}"
TOP_P="${TOP_P:-0}"          # top_k 和 top_p 不能同时 > 0

echo "=== Evo2 Sequence Generation ==="
echo "Checkpoint: ${CKPT_DIR}"
echo "Prompt: ${PROMPT:0:50}..."
echo "Max new tokens: ${MAX_NEW_TOKENS}"

infer_evo2 \
    --ckpt-dir "${CKPT_DIR}" \
    --prompt "${PROMPT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-k "${TOP_K}" \
    --top-p "${TOP_P}"
