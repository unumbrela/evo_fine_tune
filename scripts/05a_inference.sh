#!/usr/bin/env bash
# Stage 5a: 使用微调后的模型进行序列生成
# 需要在 BioNeMo 容器内运行
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 微调后的 checkpoint 目录 (train_evo2 的输出)
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/results/evo2_1b_ft/checkpoints}"

# 输入 prompt (DNA 序列片段)
PROMPT="${PROMPT:-ATGAAAGCAATTTTCGTACTGAAAATTTCGATTCTTACCTTAGCAGGCG}"

# 生成参数
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-500}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-4}"
TOP_P="${TOP_P:-0.95}"

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
