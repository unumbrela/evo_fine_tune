#!/usr/bin/env bash
# 从 HuggingFace 下载 Savanna 格式 checkpoint 并转换为 NeMo2 格式
# 适用于: 没有 NGC 访问权限，或想使用 Savanna 原始权重
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${PROJECT_DIR}/checkpoints"
OUTPUT_DIR="${CKPT_DIR}/nemo2_evo2_1b_8k"

mkdir -p "${CKPT_DIR}"

echo "=== Converting Savanna checkpoint to NeMo2 format ==="

# 使用 hf:// URI 前缀直接从 HuggingFace 下载并转换
# 注意: 不要手动下载后再转换，可能会遇到 KeyError: 'module' 的问题
evo2_convert_to_nemo2 \
    --model-path "hf://arcinstitute/savanna_evo2_1b_base" \
    --model-size 1b \
    --output-dir "${OUTPUT_DIR}"

echo "=== Conversion complete ==="
echo "NeMo2 checkpoint: ${OUTPUT_DIR}"
echo ""
echo "注意: 此 checkpoint 为 FP8 训练的权重。"
echo "如果你的 GPU 不支持 FP8，需要进行 BF16 微调 (参考 NVIDIA 教程)。"
echo "微调到 loss ~1.08 即可恢复 BF16 精度。"
