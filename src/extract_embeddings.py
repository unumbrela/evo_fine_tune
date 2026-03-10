"""
Embedding 提取脚本

从微调后的 Evo2 模型中提取序列 embedding。
参考 HPI-Potsdam 的实现:
  - 使用中间层 (而非最后几层) 的 embedding
  - Mean pooling 聚合 token-level embedding 为 sequence-level
  - 输出 4096 维向量

用法:
    python src/extract_embeddings.py \\
        --fasta data/raw/sequences.fasta \\
        --ckpt-dir results/evo2_1b_ft/checkpoints/ \\
        --output results/embeddings/embeddings.npz \\
        --layer 16 \\
        --batch-size 4

    # 使用 BioNeMo 内置的 predict_evo2 (仅输出 log-prob):
    # predict_evo2 --fasta ... --ckpt-dir ... --output-dir ... --output-log-prob-seqs
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Evo2 embeddings from FASTA sequences")
    parser.add_argument("--fasta", type=Path, required=True, help="Input FASTA file")
    parser.add_argument("--ckpt-dir", type=Path, required=True, help="Model checkpoint directory")
    parser.add_argument("--output", type=Path, default=Path("results/embeddings/embeddings.npz"))
    parser.add_argument("--layer", type=int, default=16,
                        help="Which model layer to extract embeddings from (HPI used layer 16)")
    parser.add_argument("--pooling", choices=["mean", "max"], default="mean",
                        help="Pooling strategy: mean (recommended) or max")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8192,
                        help="Max sequence length (truncate longer sequences)")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    return parser.parse_args()


def read_fasta(fasta_path: Path):
    """Simple FASTA parser. Yields (header, sequence) tuples."""
    header = None
    seq_lines = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:]
                seq_lines = []
            elif line:
                seq_lines.append(line.upper())

    if header is not None:
        yield header, "".join(seq_lines)


def extract_embeddings_bionemo(fasta_path, ckpt_dir, layer, pooling, batch_size, max_length, device):
    """
    Extract embeddings using BioNeMo's model loading infrastructure.

    This hooks into intermediate layers of the Hyena model to capture
    activation vectors, then pools across the sequence dimension.
    """
    try:
        import torch
        from bionemo.evo2.data.tokenizer import Evo2Tokenizer
    except ImportError:
        print("ERROR: bionemo-evo2 not installed.")
        print("Run inside BioNeMo container or: pip install bionemo-evo2")
        sys.exit(1)

    # Load model
    print(f"Loading model from {ckpt_dir}...")
    # NOTE: The exact loading API depends on your BioNeMo version.
    # This is a reference implementation. Adjust based on your setup.
    #
    # 方式 1: 使用 evo2 包的 inference API (推荐)
    try:
        from evo2 import Evo2
        model = Evo2("evo2-1b", checkpoint_path=str(ckpt_dir))
        use_evo2_api = True
        print("Using evo2 inference API")
    except (ImportError, Exception):
        use_evo2_api = False
        print("evo2 package not available, attempting BioNeMo model loading...")

        # 方式 2: 使用 BioNeMo 的 Megatron 模型加载
        # 这需要更复杂的设置，参考 HPI-Potsdam 的 embed_evo2 脚本
        print("WARNING: Full BioNeMo model loading requires Megatron initialization.")
        print("Consider using the evo2 package directly: pip install evo2")
        print("Or use predict_evo2 CLI for log-probability scoring.")
        sys.exit(1)

    # Read sequences
    sequences = []
    headers = []
    for header, seq in read_fasta(fasta_path):
        if len(seq) > max_length:
            seq = seq[:max_length]
        sequences.append(seq)
        headers.append(header)

    print(f"Loaded {len(sequences)} sequences")

    # Extract embeddings
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        batch_headers = headers[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(sequences)-1)//batch_size + 1}...")

        if use_evo2_api:
            # evo2 package: embed sequences
            for seq in batch:
                # The evo2 package provides embedding extraction
                # Exact API may vary; this is the general pattern
                with torch.no_grad():
                    emb = model.embed(seq, layer=layer)
                    # emb shape: (seq_len, hidden_dim=4096)

                    if pooling == "mean":
                        emb = emb.mean(dim=0)  # (4096,)
                    elif pooling == "max":
                        emb = emb.max(dim=0).values

                    embeddings.append(emb.cpu().numpy())

    embeddings = np.stack(embeddings)  # (N, 4096)
    print(f"Embeddings shape: {embeddings.shape}")

    return headers, embeddings


def main():
    args = parse_args()

    if not args.fasta.exists():
        print(f"ERROR: FASTA file not found: {args.fasta}")
        sys.exit(1)

    if not args.ckpt_dir.exists():
        print(f"ERROR: Checkpoint not found: {args.ckpt_dir}")
        sys.exit(1)

    headers, embeddings = extract_embeddings_bionemo(
        fasta_path=args.fasta,
        ckpt_dir=args.ckpt_dir,
        layer=args.layer,
        pooling=args.pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        headers=np.array(headers, dtype=object),
    )
    print(f"Saved embeddings to {args.output}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  File size: {args.output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
