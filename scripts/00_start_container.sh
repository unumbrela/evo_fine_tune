#!/usr/bin/env bash
# 启动 NVIDIA BioNeMo Framework Docker 容器
# 容器内包含 bionemo-evo2、preprocess_evo2、train_evo2 等所有工具
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# BioNeMo Framework 版本 (推荐 2.6+)
BIONEMO_IMAGE="${BIONEMO_IMAGE:-nvcr.io/nvidia/clara/bionemo-framework:2.6.3}"

echo "=== Starting BioNeMo container ==="
echo "Project dir: ${PROJECT_DIR}"
echo "Image: ${BIONEMO_IMAGE}"

docker run --rm -it \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${PROJECT_DIR}:/workspace/evo_fine_tune" \
    -w /workspace/evo_fine_tune \
    -e HF_HOME=/workspace/evo_fine_tune/.cache/huggingface \
    "${BIONEMO_IMAGE}" \
    /bin/bash
