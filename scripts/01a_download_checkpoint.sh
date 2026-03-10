#!/usr/bin/env bash
# 下载 Evo2 1B checkpoint (NeMo2 格式，从 NGC)
# 需要在 BioNeMo 容器内运行，或已安装 bionemo-core
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${PROJECT_DIR}/checkpoints"
mkdir -p "${CKPT_DIR}"

echo "=== Downloading Evo2 1B checkpoint ==="

# 选择 checkpoint:
#   evo2/1b-8k:1.0       -> FP8 版本 (需要 H100)
#   evo2/1b-8k-bf16:1.0  -> BF16 版本 (A100/A6000 可用)
#
# 如果你的 GPU 不支持 FP8 (非 Hopper 架构)，请使用 bf16 版本:

MODEL_ID="${MODEL_ID:-evo2/1b-8k-bf16:1.0}"

echo "Downloading model: ${MODEL_ID}"
echo "Target dir: ${CKPT_DIR}"

# BioNeMo 内置下载工具
python -c "
from bionemo.core.data.load import load
ckpt_path = load('${MODEL_ID}')
print(f'Checkpoint downloaded to: {ckpt_path}')
"

echo "=== Download complete ==="
echo "Checkpoint location: ${CKPT_DIR}"
