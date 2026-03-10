# Evo2 BioNeMo Fine-Tuning Pipeline

基于 NVIDIA BioNeMo Framework 的 Evo2 1B 微调项目模板，覆盖基因序列数据准备、预处理、训练、推理、embedding 提取和评估的完整流程。

这个仓库的目标不是重新实现 Evo2，而是把官方工具链和实际实验流程整理成一套可复用、可扩展、适合实验室环境落地的工作流。

## 项目定位

- 面向 Evo2 1B 的 DNA / genomics 序列微调
- 依赖 BioNeMo 提供的 `preprocess_evo2`、`train_evo2`、`infer_evo2` 等工具
- 支持官方 OpenGenome2 示例数据、iGEM Registry 数据和自定义 FASTA
- 提供全量微调与 LoRA 微调两套训练入口
- 提供 embedding 提取与基础评估脚本，便于下游分析

## 项目mu'lu

```text
evo_fine_tune/
  configs/
    preprocessing.yaml
    dataset_blend.yaml
  scripts/
    00_start_container.sh
    01a_download_checkpoint.sh
    01b_convert_checkpoint.sh
    02a_download_opengenome2_sample.sh
    03_preprocess.sh
    04_train.sh
    04_train_lora.sh
    04_train.slurm
    05a_inference.sh
  src/
    prepare_igem_data.py
    dedup_with_mmseqs2.py
    extract_embeddings.py
    evaluate.py
  data/
```

## Pipeline 概览

```text
Stage 0  环境搭建 + 容器启动
Stage 1  Checkpoint 下载 / 转换
Stage 2  数据准备与清洗
Stage 3  FASTA -> Megatron IndexedDataset
Stage 4  Evo2 1B 微调 / LoRA 微调
Stage 5  生成推理 / embedding 提取
Stage 6  序列统计与 embedding 评估
```

## 适合什么场景

- 复现实验室内部的 Evo2 微调基线
- 将合成生物学、基因组学或 iGEM 相关序列整理成训练数据
- 在 BioNeMo 容器环境中快速跑通从数据到结果的最短路径
- 为后续的检索、聚类、可视化或下游预测任务产出 embedding

## 快速开始

### 0. 启动 BioNeMo 容器

```bash
bash scripts/00_start_container.sh
```

后续命令默认在容器内的 `/workspace/evo_fine_tune` 路径执行。

### 1. 下载或转换 checkpoint

```bash
# 推荐：直接下载可用的 NeMo2 checkpoint
bash scripts/01a_download_checkpoint.sh

# 备选：从 HuggingFace 的 Savanna checkpoint 转换
bash scripts/01b_convert_checkpoint.sh
```

### 2. 准备训练数据

```bash
# OpenGenome2 示例数据
bash scripts/02a_download_opengenome2_sample.sh

# iGEM XML dump -> FASTA
python src/prepare_igem_data.py \
  --xml-dump data/raw/igem_dump.xml \
  --output data/raw/igem_sequences.fasta \
  --min-length 100 \
  --composite-only
```

如果你的数据存在明显近重复，建议在训练前先做 MMseqs2 聚类去重：

```bash
python src/dedup_with_mmseqs2.py \
  --input data/raw/igem_sequences.fasta \
  --output-dir data/clustered
```

### 3. 预处理

```bash
bash scripts/03_preprocess.sh
```

默认使用 [`configs/preprocessing.yaml`](configs/preprocessing.yaml)，将 FASTA 转换成 BioNeMo / Megatron 可消费的 `.bin` 和 `.idx` 文件。

### 4. 微调

```bash
# 单卡全量微调
bash scripts/04_train.sh

# LoRA 微调
bash scripts/04_train_lora.sh

# 多卡 / 集群
sbatch scripts/04_train.slurm
```

### 5. 推理与 embedding

```bash
# 生成序列
bash scripts/05a_inference.sh

# 提取 sequence embedding
python src/extract_embeddings.py \
  --fasta data/raw/sequences.fasta \
  --ckpt-dir results/evo2_1b_ft/checkpoints \
  --output results/embeddings/embeddings.npz
```

### 6. 评估

```bash
python src/evaluate.py \
  --embeddings results/embeddings/embeddings.npz \
  --metadata data/raw/metadata.csv \
  --output-dir results/evaluation
```

## 核心脚本说明

- [`scripts/03_preprocess.sh`](scripts/03_preprocess.sh): 调用 `preprocess_evo2` 生成训练/验证/测试集二进制文件
- [`scripts/04_train.sh`](scripts/04_train.sh): 全量微调入口，支持通过环境变量覆写训练参数
- [`scripts/04_train_lora.sh`](scripts/04_train_lora.sh): 更低显存占用的 LoRA 微调入口
- [`src/prepare_igem_data.py`](src/prepare_igem_data.py): 从 iGEM Registry XML 提取、清洗并去重 DNA 序列
- [`src/dedup_with_mmseqs2.py`](src/dedup_with_mmseqs2.py): 基于 MMseqs2 的近重复聚类和数据拆分
- [`src/extract_embeddings.py`](src/extract_embeddings.py): 从微调后 Evo2 模型导出 sequence embedding
- [`src/evaluate.py`](src/evaluate.py): 评估 GC 分布、k-mer 差异、embedding 距离和 KNN 对齐表现

## 运行前提

- NVIDIA GPU
- Docker + `nvidia-container-toolkit`，或可用的 BioNeMo 运行环境
- BioNeMo / Evo2 对应版本工具链
- 足够的本地磁盘空间保存 checkpoint、缓存和训练结果

## 硬件建议

| 配置 | 最低要求 | 推荐 |
|------|---------|------|
| GPU 显存 | 45 GB | 80 GB |
| BF16 微调 | A100 / A6000 / L40S | A100 80GB |
| FP8 微调 | H100 | H100 |
| LoRA 微调 | 15-24 GB | 24+ GB |
| 系统内存 | 32 GB | 64 GB |
| 磁盘 | 50 GB | 200 GB |

说明：

- 原始 1B checkpoint 训练配置偏向 FP8，非 H100 环境更适合使用 BF16 版本或 LoRA
- `configs/*.yaml` 中当前路径按容器内 `/workspace/evo_fine_tune` 编写

## 参考资料

- NVIDIA BioNeMo Evo2 fine-tuning tutorial
- Arc Institute Evo2 repository
- HPI-Potsdam iGEM 2025 project notes
- `savanna_evo2_1b_base` HuggingFace checkpoint
