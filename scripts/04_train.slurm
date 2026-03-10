#!/bin/bash
#SBATCH --job-name=evo2-1b-ft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/evo2-ft-%j.out
#SBATCH --error=logs/evo2-ft-%j.err

# ============================================================
# SLURM 训练脚本
# 用法: sbatch scripts/04_train.slurm
#
# 根据你的集群调整:
#   --gres=gpu:N        GPU 数量
#   --partition=xxx     分区名
#   --account=xxx       账户
#   --time=HH:MM:SS     时间限制
# ============================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# --- 集群配置 (根据你的环境修改) ---
# 如果集群使用 enroot/pyxis 容器:
# CONTAINER="nvcr.io/nvidia/clara/bionemo-framework:2.6.3"
# srun --container-image="${CONTAINER}" --container-mounts="$PWD:/workspace/evo_fine_tune" \
#     bash /workspace/evo_fine_tune/scripts/04_train.sh

# 如果集群使用 singularity:
# CONTAINER="/path/to/bionemo-framework-2.6.3.sif"
# srun singularity exec --nv --bind "$PWD:/workspace/evo_fine_tune" "${CONTAINER}" \
#     bash /workspace/evo_fine_tune/scripts/04_train.sh

# 如果已经在容器内或已 pip install bionemo-evo2:
export DEVICES="${SLURM_GPUS_ON_NODE:-1}"
export NUM_NODES="${SLURM_NNODES:-1}"

# --- 训练参数 ---
export MAX_STEPS="${MAX_STEPS:-5000}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export LR="${LR:-3e-4}"
export USE_FP8="${USE_FP8:-false}"
export USE_WANDB="${USE_WANDB:-false}"

echo "=== SLURM Job ${SLURM_JOB_ID} ==="
echo "Node: $(hostname)"
echo "GPUs: ${DEVICES}"
echo "Nodes: ${NUM_NODES}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "============================================================"

srun bash scripts/04_train.sh
