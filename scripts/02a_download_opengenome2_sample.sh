#!/usr/bin/env bash
# 下载 OpenGenome2 样本数据用于测试微调流程
# OpenGenome2 是 Evo2 预训练数据集，包含 8.8T tokens
# 这里只下载少量样本用于验证 pipeline
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${PROJECT_DIR}/data/raw"
mkdir -p "${RAW_DIR}"

echo "=== Downloading OpenGenome2 sample data ==="

# --- 方式 1: 从 HuggingFace 下载 OpenGenome2 样本 ---
# OpenGenome2 完整数据集: https://huggingface.co/datasets/arcinstitute/opengenome2
# 数据格式: FASTA (raw) 或 JSONL (preprocessed)
#
# 由于完整数据集非常大 (~数 TB)，这里只下载一个小子集用于测试
# 你可以用 huggingface-cli 或 Python 脚本选择性下载

python3 << 'PYEOF'
import os
import subprocess
import sys

raw_dir = os.environ.get("RAW_DIR", "data/raw")
sample_fasta = os.path.join(raw_dir, "opengenome2_sample.fasta")

if os.path.exists(sample_fasta):
    print(f"Sample data already exists: {sample_fasta}")
    sys.exit(0)

print("Attempting to download OpenGenome2 sample from HuggingFace...")
print("Note: Full dataset is very large. Downloading a small sample only.")

try:
    # 尝试使用 datasets 库下载少量样本
    from datasets import load_dataset

    # 加载少量原核基因组数据 (streaming 模式避免下载全部)
    ds = load_dataset(
        "arcinstitute/opengenome2",
        split="train",
        streaming=True,
    )

    count = 0
    max_sequences = 500  # 下载 500 条序列用于测试
    with open(sample_fasta, "w") as f:
        for record in ds:
            # OpenGenome2 JSONL 格式: {"text": "ACGT...", "id": "..."}
            seq = record.get("text", record.get("sequence", ""))
            seq_id = record.get("id", record.get("name", f"seq_{count}"))
            if seq and len(seq) >= 100:  # 至少 100bp
                f.write(f">{seq_id}\n{seq}\n")
                count += 1
                if count >= max_sequences:
                    break

    print(f"Downloaded {count} sequences to {sample_fasta}")

except ImportError:
    print("'datasets' library not available. Creating synthetic test data instead.")
    print("Install with: pip install datasets")
    print("")
    print("Generating minimal test FASTA for pipeline validation...")

    import random
    random.seed(42)
    bases = "ACGT"

    with open(sample_fasta, "w") as f:
        for i in range(100):
            length = random.randint(500, 5000)
            seq = "".join(random.choice(bases) for _ in range(length))
            f.write(f">synthetic_seq_{i}\n{seq}\n")

    print(f"Generated 100 synthetic sequences to {sample_fasta}")
    print("WARNING: Synthetic data is for pipeline testing only, not meaningful fine-tuning.")

except Exception as e:
    print(f"Error downloading: {e}")
    print("You can manually download from: https://huggingface.co/datasets/arcinstitute/opengenome2")
    sys.exit(1)

PYEOF

echo "=== Data preparation complete ==="
echo "Data location: ${RAW_DIR}"
echo ""
echo "如果需要使用更多数据:"
echo "  1. OpenGenome2 完整数据: https://huggingface.co/datasets/arcinstitute/opengenome2"
echo "  2. iGEM Registry: python src/prepare_igem_data.py"
echo "  3. 自定义 FASTA: 直接放到 data/raw/ 目录"
