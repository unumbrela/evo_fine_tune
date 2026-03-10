"""
Evo2 微调评估脚本

参考 HPI-Potsdam 的评估方法:
  1. 序列层面: GC content 分布比较, k-mer JSD
  2. Embedding 空间: cosine distance 分布, UMAP 可视化
  3. 检索评估: nearest-neighbor part type 混淆矩阵

用法:
    # 基础评估 (只需要 embedding 文件)
    python src/evaluate.py \\
        --embeddings results/embeddings/embeddings.npz \\
        --output-dir results/evaluation/

    # 完整评估 (需要 metadata 和生成序列)
    python src/evaluate.py \\
        --embeddings results/embeddings/embeddings.npz \\
        --generated-fasta results/generated/generated.fasta \\
        --original-fasta data/raw/test_sequences.fasta \\
        --metadata data/raw/metadata.csv \\
        --output-dir results/evaluation/
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Evo2 fine-tuning results")
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Embeddings .npz file from extract_embeddings.py")
    parser.add_argument("--generated-fasta", type=Path, default=None,
                        help="FASTA of generated sequences (for sequence-level metrics)")
    parser.add_argument("--original-fasta", type=Path, default=None,
                        help="FASTA of original sequences being replaced")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="CSV with columns: header, part_type (for retrieval eval)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors for retrieval")
    return parser.parse_args()


def read_fasta_simple(path: Path):
    """Simple FASTA parser."""
    sequences = []
    headers = []
    header = None
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    sequences.append("".join(seq_lines))
                    headers.append(header)
                header = line[1:]
                seq_lines = []
            elif line:
                seq_lines.append(line.upper())
    if header is not None:
        sequences.append("".join(seq_lines))
        headers.append(header)
    return headers, sequences


# ---- Sequence-level metrics ----

def gc_content(seq: str) -> float:
    """Calculate GC content of a DNA sequence."""
    if len(seq) == 0:
        return 0.0
    gc = sum(1 for b in seq if b in "GC")
    return gc / len(seq)


def kmer_distribution(seq: str, k: int) -> dict:
    """Calculate k-mer frequency distribution."""
    counts = Counter()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        counts[kmer] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {kmer: count / total for kmer, count in counts.items()}


def jensen_shannon_divergence(p_dict: dict, q_dict: dict) -> float:
    """Calculate Jensen-Shannon divergence between two distributions."""
    all_keys = set(p_dict.keys()) | set(q_dict.keys())
    p = np.array([p_dict.get(k, 0.0) for k in all_keys])
    q = np.array([q_dict.get(k, 0.0) for k in all_keys])

    # Normalize
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)

    m = 0.5 * (p + q)

    # KL divergence with numerical stability
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / (b[mask] + 1e-12)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def evaluate_sequences(generated_seqs, original_seqs):
    """Evaluate generated vs original sequences (HPI-Potsdam metrics)."""
    print("\n=== Sequence-Level Evaluation ===")

    # GC content
    gen_gc = [gc_content(s) for s in generated_seqs]
    orig_gc = [gc_content(s) for s in original_seqs]

    print(f"GC content - Generated: {np.mean(gen_gc):.4f} +/- {np.std(gen_gc):.4f}")
    print(f"GC content - Original:  {np.mean(orig_gc):.4f} +/- {np.std(orig_gc):.4f}")

    # Wasserstein distance for GC content
    try:
        from scipy.stats import wasserstein_distance, ks_2samp
        wd = wasserstein_distance(gen_gc, orig_gc)
        ks_stat, ks_pval = ks_2samp(gen_gc, orig_gc)
        print(f"GC Wasserstein distance: {wd:.4f}")
        print(f"GC KS test p-value: {ks_pval:.6f}")
    except ImportError:
        print("(Install scipy for Wasserstein distance and KS test)")

    # k-mer JSD
    for k in [2, 3, 4, 5]:
        gen_kmer = Counter()
        orig_kmer = Counter()
        for s in generated_seqs:
            gen_kmer.update(kmer_distribution(s, k))
        for s in original_seqs:
            orig_kmer.update(kmer_distribution(s, k))

        # Normalize
        gen_total = sum(gen_kmer.values())
        orig_total = sum(orig_kmer.values())
        gen_dist = {kk: v / gen_total for kk, v in gen_kmer.items()}
        orig_dist = {kk: v / orig_total for kk, v in orig_kmer.items()}

        jsd = jensen_shannon_divergence(gen_dist, orig_dist)
        print(f"{k}-mer JSD: {jsd:.6f}")

    return {"gen_gc": gen_gc, "orig_gc": orig_gc}


# ---- Embedding-level metrics ----

def cosine_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distances between rows of a and b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    similarity = a_norm @ b_norm.T
    return 1.0 - similarity


def evaluate_embeddings(embeddings: np.ndarray, headers: list, k: int = 10,
                        metadata_path: Path | None = None):
    """Evaluate embedding space quality."""
    print("\n=== Embedding Space Evaluation ===")
    print(f"Embeddings shape: {embeddings.shape}")

    # Basic statistics
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Embedding norms - mean: {norms.mean():.2f}, std: {norms.std():.2f}")

    # Pairwise cosine distances (sample if too many)
    n = len(embeddings)
    if n > 2000:
        idx = np.random.choice(n, 2000, replace=False)
        sample_emb = embeddings[idx]
    else:
        sample_emb = embeddings

    dist_matrix = cosine_distance_matrix(sample_emb, sample_emb)
    # Exclude self-distances
    mask = ~np.eye(len(sample_emb), dtype=bool)
    pairwise_dists = dist_matrix[mask]

    print(f"Pairwise cosine distance - mean: {pairwise_dists.mean():.4f}, "
          f"std: {pairwise_dists.std():.4f}")
    print(f"  min: {pairwise_dists.min():.4f}, max: {pairwise_dists.max():.4f}")

    # KNN retrieval evaluation (if metadata available)
    if metadata_path is not None:
        try:
            import pandas as pd
            meta = pd.read_csv(metadata_path)

            # Build header -> part_type mapping
            type_map = {}
            for _, row in meta.iterrows():
                type_map[str(row.get("header", row.get("name", "")))] = str(
                    row.get("part_type", row.get("type", "unknown"))
                )

            # Evaluate part type alignment in KNN
            correct = 0
            total = 0
            for i in range(min(n, 1000)):
                query_type = type_map.get(headers[i], None)
                if query_type is None:
                    continue

                # Find k nearest neighbors
                dists = cosine_distance_matrix(embeddings[i:i+1], embeddings)[0]
                dists[i] = float("inf")  # exclude self
                nn_idx = np.argsort(dists)[:k]

                nn_types = [type_map.get(headers[j], "unknown") for j in nn_idx]
                matches = sum(1 for t in nn_types if t == query_type)
                correct += matches
                total += k

            if total > 0:
                print(f"\nKNN part type alignment (k={k}): {correct/total:.4f}")
                print(f"  ({correct}/{total} neighbors match query part type)")

        except (ImportError, FileNotFoundError, KeyError) as e:
            print(f"Skipping KNN eval: {e}")

    # UMAP visualization
    try:
        import umap

        print("\nGenerating UMAP projection...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
        embedding_2d = reducer.fit_transform(sample_emb)

        # Save UMAP coordinates
        np.savez_compressed(
            "results/evaluation/umap_coords.npz",
            coords=embedding_2d,
        )
        print(f"UMAP coordinates saved ({embedding_2d.shape})")

    except ImportError:
        print("(Install umap-learn for UMAP visualization: pip install umap-learn)")

    return {"pairwise_dists": pairwise_dists}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    data = np.load(args.embeddings, allow_pickle=True)
    embeddings = data["embeddings"]
    headers = list(data["headers"])

    # Embedding evaluation
    emb_results = evaluate_embeddings(
        embeddings, headers, k=args.k, metadata_path=args.metadata
    )

    # Sequence-level evaluation (optional)
    if args.generated_fasta and args.original_fasta:
        _, gen_seqs = read_fasta_simple(args.generated_fasta)
        _, orig_seqs = read_fasta_simple(args.original_fasta)
        seq_results = evaluate_sequences(gen_seqs, orig_seqs)
    else:
        print("\n(Skipping sequence-level eval: provide --generated-fasta and --original-fasta)")

    print(f"\n=== Evaluation complete. Results in {args.output_dir} ===")


if __name__ == "__main__":
    main()
