# Evo2 Fine-Tuning Pipeline

End-to-end pipeline for fine-tuning [Evo2](https://arcinstitute.org/tools/evo/evo2) (1B) on custom DNA sequences using NVIDIA [BioNeMo Framework](https://docs.nvidia.com/bionemo-framework/). Covers data preparation, preprocessing, LoRA / full fine-tuning, sequence generation, embedding extraction, and evaluation.

## Pipeline Overview

```
Stage 0  Environment setup & container launch
Stage 1  Checkpoint download / conversion
Stage 2  Data preparation & cleaning (iGEM / custom FASTA)
Stage 3  FASTA → Megatron IndexedDataset (.bin/.idx)
Stage 4  Fine-tuning (full or LoRA)
Stage 5  Sequence generation & embedding extraction
Stage 6  Evaluation & visualization
```

## Project Structure

```
evo_fine_tune/
├── configs/
│   ├── preprocessing.yaml        # FASTA → binary preprocessing config
│   └── dataset_blend.yaml        # Training dataset blend config
├── scripts/
│   ├── 00_start_container.sh     # Launch BioNeMo container
│   ├── 01a_download_checkpoint.sh
│   ├── 01b_convert_checkpoint.sh # HuggingFace → NeMo2 conversion
│   ├── 02a_download_opengenome2_sample.sh
│   ├── 03_preprocess.sh          # FASTA → Megatron .bin/.idx
│   ├── 04_train.sh               # Full fine-tuning
│   ├── 04_train_lora.sh          # LoRA fine-tuning (~1.5% params)
│   ├── 04_train.slurm            # Multi-node SLURM job
│   └── 05a_inference.sh          # Sequence generation
├── src/
│   ├── prepare_igem_data.py      # iGEM XML dump → FASTA
│   ├── clean_igem_fasta.py       # FASTA cleaning & filtering
│   ├── dedup_with_mmseqs2.py     # MMseqs2 clustering & dedup
│   ├── extract_embeddings.py     # Extract sequence embeddings via forward hooks
│   ├── evaluate.py               # Embedding space & sequence metrics
│   └── visualize_embeddings.py   # UMAP plots, distance analysis, report
├── data/                         # Training data (git-ignored)
├── checkpoints/                  # Model checkpoints (git-ignored)
└── results/                      # Training & evaluation outputs (git-ignored)
```

## Quick Start

### 0. Environment Setup

**Option A: BioNeMo Container (recommended)**

```bash
bash scripts/00_start_container.sh
```

**Option B: Manual installation**

```bash
git clone https://github.com/NVIDIA/bionemo-framework.git
cd bionemo-framework
pip install -e sub-packages/bionemo-noodles
pip install -e sub-packages/bionemo-core
pip install -e sub-packages/bionemo-llm
pip install -e sub-packages/bionemo-evo2
```

> If using PyTorch 2.8 instead of the container's default, you may need to source-build TransformerEngine with `--no-build-isolation` and install the bundled NeMo/Megatron from `bionemo-framework/3rdparty/`.

### 1. Download Checkpoint

```bash
# NeMo2 checkpoint (recommended)
bash scripts/01a_download_checkpoint.sh

# Or convert from HuggingFace
bash scripts/01b_convert_checkpoint.sh
```

### 2. Prepare Data

```bash
# Option A: OpenGenome2 sample data
bash scripts/02a_download_opengenome2_sample.sh

# Option B: iGEM data
python src/prepare_igem_data.py \
  --xml-dump data/raw/igem_dump.xml \
  --output data/raw/igem_sequences.fasta \
  --min-length 100

# Clean & filter
python src/clean_igem_fasta.py \
  --input data/raw/igem_sequences.fasta \
  --output data/raw/igem_cleaned.fasta \
  --min-length 100

# Optional: MMseqs2 deduplication
python src/dedup_with_mmseqs2.py \
  --input data/raw/igem_cleaned.fasta \
  --output-dir data/clustered
```

### 3. Preprocess

```bash
bash scripts/03_preprocess.sh
```

Converts FASTA into Megatron-compatible `.bin` / `.idx` files. Configure via [`configs/preprocessing.yaml`](configs/preprocessing.yaml).

### 4. Fine-Tune

```bash
# Full fine-tuning (requires ~45GB+ VRAM)
bash scripts/04_train.sh

# LoRA fine-tuning (~16.8M params, ~15GB VRAM)
bash scripts/04_train_lora.sh

# Multi-node (SLURM)
sbatch scripts/04_train.slurm
```

All training hyperparameters can be overridden via environment variables (see script headers).

### 5. Inference & Embedding Extraction

```bash
# Generate sequences
bash scripts/05a_inference.sh

# Extract embeddings (base model)
python src/extract_embeddings.py \
  --fasta data/raw/igem_sequences.fasta \
  --ckpt-dir <BASE_CHECKPOINT_DIR> \
  --output results/embeddings/embeddings.npz

# Extract embeddings (with LoRA adapter)
python src/extract_embeddings.py \
  --fasta data/raw/igem_sequences.fasta \
  --ckpt-dir <BASE_CHECKPOINT_DIR> \
  --lora-checkpoint-path <LORA_CHECKPOINT_DIR> \
  --output results/embeddings/embeddings.npz \
  --max-length 2048
```

### 6. Evaluation & Visualization

```bash
# Quantitative evaluation (embedding stats, cosine distance, UMAP, KNN)
python src/evaluate.py \
  --embeddings results/embeddings/embeddings.npz \
  --output-dir results/evaluation

# Visualization (generates PNG plots + report)
python src/visualize_embeddings.py \
  --embeddings results/embeddings/embeddings.npz \
  --output-dir results/evaluation
```

**Generated outputs:**

| File | Description |
|------|-------------|
| `umap_by_type.png` | UMAP projection colored by part type |
| `cosine_distance_distribution.png` | Intra-class vs inter-class distance |
| `part_type_distribution.png` | Part type frequency distribution |
| `embedding_norms.png` | Embedding L2 norm distribution |
| `length_vs_norm.png` | Sequence length vs embedding norm |
| `report.txt` | Summary statistics |

## Hardware Requirements

| Setup | VRAM | GPU Examples |
|-------|------|-------------|
| LoRA fine-tuning | 15-24 GB | RTX 3090, RTX 4090, RTX 5070 Ti |
| BF16 full fine-tuning | 45-80 GB | A100, A6000, L40S |
| FP8 full fine-tuning | 40+ GB | H100 |

## Key Dependencies

- [NVIDIA BioNeMo Framework](https://docs.nvidia.com/bionemo-framework/) (evo2, llm, core)
- NeMo Toolkit + Megatron-Core
- PyTorch 2.8+ with CUDA
- TransformerEngine

## References

- [NVIDIA BioNeMo Evo2 Documentation](https://docs.nvidia.com/bionemo-framework/)
- [Arc Institute Evo2](https://arcinstitute.org/tools/evo/evo2)
- [Evo2 GitHub](https://github.com/ArcInstitute/evo2)
- [HPI-Potsdam iGEM 2025 Project](https://2025.igem.wiki/hpi-potsdam/)
