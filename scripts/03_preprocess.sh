#!/usr/bin/env bash
# Stage 3: 预处理 - 将 FASTA 转换为 Megatron 二进制格式
# 需要在 BioNeMo 容器内运行
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${PROJECT_DIR}/configs/preprocessing.yaml}"

echo "=== Stage 3: Preprocessing FASTA -> binary ==="
echo "Config: ${CONFIG}"

# 确保输出目录存在
mkdir -p "${PROJECT_DIR}/data/processed"

# 运行 BioNeMo 预处理器
# 输入: FASTA 文件
# 输出: .bin + .idx 文件 (Megatron IndexedDataset 格式)
preprocess_evo2 -c "${CONFIG}"

echo ""
echo "=== Preprocessing complete ==="
echo "Output files:"
ls -la "${PROJECT_DIR}/data/processed/"
echo ""
echo "生成的文件说明:"
echo "  *_train_text_CharLevelTokenizer_document.bin/idx  -> 训练集"
echo "  *_valid_text_CharLevelTokenizer_document.bin/idx  -> 验证集"
echo "  *_test_text_CharLevelTokenizer_document.bin/idx   -> 测试集"
echo ""
echo "下一步: 确认 configs/dataset_blend.yaml 中的 dataset_prefix 路径正确"
echo "然后运行: bash scripts/04_train.sh"
