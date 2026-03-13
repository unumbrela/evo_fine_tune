"""
Embedding 可视化与量化分析脚本

生成:
  1. UMAP 散点图 (按 part type 着色)
  2. Part type 分布柱状图
  3. Embedding 范数分布直方图
  4. 类内/类间 cosine distance 分布对比
  5. 量化统计报告 (终端输出 + txt 保存)

用法:
    python src/visualize_embeddings.py \
        --embeddings results/embeddings/embeddings.npz \
        --output-dir results/evaluation
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Evo2 embeddings")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--umap-coords", type=Path, default=None,
                        help="Pre-computed UMAP coordinates (.npz). Auto-detected if not set.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--top-types", type=int, default=8,
                        help="Show top N part types in plots (rest grouped as 'other')")
    return parser.parse_args()


def extract_part_type(header: str) -> str:
    """Extract part type from header like '79635 name=BBa_K1529987 type=regulatory length=92bp'."""
    m = re.search(r"type=(\S+)", header)
    return m.group(1) if m else "unknown"


def extract_length(header: str) -> int:
    """Extract sequence length from header."""
    m = re.search(r"length=(\d+)", header)
    return int(m.group(1)) if m else 0


def cosine_sim(a, b):
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_n @ b_n.T


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = np.load(args.embeddings, allow_pickle=True)
    embeddings = data["embeddings"]
    headers = list(data["headers"])

    # Extract metadata from headers
    part_types = [extract_part_type(h) for h in headers]
    seq_lengths = [extract_length(h) for h in headers]

    # Count part types
    type_counts = Counter(part_types)
    top_types = [t for t, _ in type_counts.most_common(args.top_types)]
    # Map rare types to 'other'
    labels = [t if t in top_types else "other" for t in part_types]
    label_set = top_types + (["other"] if "other" in labels else [])

    print(f"Total sequences: {len(embeddings)}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"\nPart type distribution:")
    for t, c in type_counts.most_common(15):
        print(f"  {t:20s}: {c:5d} ({c/len(embeddings)*100:.1f}%)")

    # ---- 1. Part type distribution bar chart ----
    fig, ax = plt.subplots(figsize=(10, 5))
    types_sorted = type_counts.most_common(15)
    ax.barh([t for t, _ in types_sorted], [c for _, c in types_sorted], color="steelblue")
    ax.set_xlabel("Count")
    ax.set_title("Part Type Distribution (iGEM)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(args.output_dir / "part_type_distribution.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: part_type_distribution.png")

    # ---- 2. Embedding norm distribution ----
    norms = np.linalg.norm(embeddings, axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(norms, bins=50, color="coral", edgecolor="white", alpha=0.8)
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Count")
    ax.set_title(f"Embedding Norm Distribution (mean={norms.mean():.1f}, std={norms.std():.1f})")
    ax.axvline(norms.mean(), color="red", linestyle="--", label=f"mean={norms.mean():.1f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(args.output_dir / "embedding_norms.png", dpi=150)
    plt.close(fig)
    print(f"Saved: embedding_norms.png")

    # ---- 3. UMAP scatter plot ----
    umap_path = args.umap_coords or (args.output_dir / "umap_coords.npz")
    if umap_path.exists():
        coords = np.load(umap_path)["coords"]
        n_umap = len(coords)
        # The UMAP was computed on a 2000-sample subset; use matching labels
        if n_umap < len(embeddings):
            # Reproduce the same random sample (seed=42 used by evaluate.py's UMAP)
            np.random.seed(42)
            idx = np.random.choice(len(embeddings), min(2000, len(embeddings)), replace=False)
            umap_labels = [labels[i] for i in idx[:n_umap]]
        else:
            umap_labels = labels[:n_umap]

        # Color map
        cmap = plt.cm.get_cmap("tab10", len(label_set))
        color_map = {t: cmap(i) for i, t in enumerate(label_set)}

        fig, ax = plt.subplots(figsize=(10, 8))
        for t in label_set:
            mask = [i for i, l in enumerate(umap_labels) if l == t]
            if mask:
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=[color_map[t]], label=f"{t} ({len(mask)})",
                           s=8, alpha=0.6)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("UMAP of Evo2 Embeddings (colored by part type)")
        ax.legend(markerscale=3, fontsize=8, loc="best")
        plt.tight_layout()
        fig.savefig(args.output_dir / "umap_by_type.png", dpi=150)
        plt.close(fig)
        print(f"Saved: umap_by_type.png")
    else:
        print("No UMAP coordinates found. Run evaluate.py first or install umap-learn.")

    # ---- 4. Intra-class vs inter-class cosine distance ----
    print("\nComputing intra/inter-class cosine distances...")
    # Sample for efficiency
    np.random.seed(0)
    sample_n = min(3000, len(embeddings))
    idx = np.random.choice(len(embeddings), sample_n, replace=False)
    sample_emb = embeddings[idx]
    sample_labels = [labels[i] for i in idx]

    sim_matrix = cosine_sim(sample_emb, sample_emb)
    dist_matrix = 1.0 - sim_matrix

    intra_dists = []
    inter_dists = []
    for i in range(sample_n):
        for j in range(i + 1, min(i + 200, sample_n)):  # limit pairs for speed
            d = dist_matrix[i, j]
            if sample_labels[i] == sample_labels[j]:
                intra_dists.append(d)
            else:
                inter_dists.append(d)

    intra_dists = np.array(intra_dists)
    inter_dists = np.array(inter_dists)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(intra_dists, bins=50, alpha=0.6, label=f"Intra-class (n={len(intra_dists)})", color="green", density=True)
    ax.hist(inter_dists, bins=50, alpha=0.6, label=f"Inter-class (n={len(inter_dists)})", color="red", density=True)
    ax.set_xlabel("Cosine Distance")
    ax.set_ylabel("Density")
    ax.set_title("Intra-class vs Inter-class Cosine Distance")
    ax.legend()
    plt.tight_layout()
    fig.savefig(args.output_dir / "cosine_distance_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved: cosine_distance_distribution.png")

    # Separation score
    if len(intra_dists) > 0 and len(inter_dists) > 0:
        separation = inter_dists.mean() - intra_dists.mean()
        print(f"\nIntra-class cosine distance: {intra_dists.mean():.4f} +/- {intra_dists.std():.4f}")
        print(f"Inter-class cosine distance: {inter_dists.mean():.4f} +/- {inter_dists.std():.4f}")
        print(f"Separation (inter - intra):  {separation:.4f}")
        if separation > 0:
            print("  -> Positive separation: embeddings cluster by part type")
        else:
            print("  -> Negative/zero separation: embeddings do NOT cluster by part type")

    # ---- 5. Sequence length vs norm ----
    if any(l > 0 for l in seq_lengths):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(seq_lengths, norms, s=3, alpha=0.3, c="steelblue")
        ax.set_xlabel("Sequence Length (bp)")
        ax.set_ylabel("Embedding L2 Norm")
        ax.set_title("Sequence Length vs Embedding Norm")
        plt.tight_layout()
        fig.savefig(args.output_dir / "length_vs_norm.png", dpi=150)
        plt.close(fig)
        print(f"Saved: length_vs_norm.png")

    # ---- Save summary report ----
    report_lines = [
        "=" * 60,
        "Evo2 Embedding Evaluation Report",
        "=" * 60,
        f"Total sequences: {len(embeddings)}",
        f"Embedding dimension: {embeddings.shape[1]}",
        f"Embedding norm: {norms.mean():.2f} +/- {norms.std():.2f}",
        "",
        "Part type distribution:",
    ]
    for t, c in type_counts.most_common(15):
        report_lines.append(f"  {t:20s}: {c:5d} ({c/len(embeddings)*100:.1f}%)")

    if len(intra_dists) > 0 and len(inter_dists) > 0:
        report_lines += [
            "",
            "Cosine distance analysis:",
            f"  Intra-class: {intra_dists.mean():.4f} +/- {intra_dists.std():.4f}",
            f"  Inter-class: {inter_dists.mean():.4f} +/- {inter_dists.std():.4f}",
            f"  Separation:  {separation:.4f}",
        ]

    report_text = "\n".join(report_lines)
    (args.output_dir / "report.txt").write_text(report_text)
    print(f"\nSaved: report.txt")
    print(f"\nAll outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()
