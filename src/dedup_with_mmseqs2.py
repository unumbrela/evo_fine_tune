"""
基于 MMseqs2 的序列去重与聚类脚本

参考 HPI-Potsdam 的流程:
  - 使用 MMseqs2 聚类去除近似重复序列
  - 最优参数: 95% identity, 90% coverage (HPI 经过 7000+ 参数组合测试)
  - 按 cluster 划分 train/val/test, 防止数据泄漏

用法:
    # 1. 安装 MMseqs2: conda install -c bioconda mmseqs2
    # 2. 运行:
    python src/dedup_with_mmseqs2.py \\
        --input data/raw/sequences.fasta \\
        --output-dir data/clustered/ \\
        --min-seq-id 0.95 \\
        --coverage 0.9 \\
        --train-ratio 0.8 \\
        --val-ratio 0.1 \\
        --test-ratio 0.1
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
import random


def parse_args():
    parser = argparse.ArgumentParser(description="MMseqs2-based deduplication and splitting")
    parser.add_argument("--input", type=Path, required=True, help="Input FASTA file")
    parser.add_argument("--output-dir", type=Path, default=Path("data/clustered"))
    parser.add_argument("--min-seq-id", type=float, default=0.95,
                        help="Minimum sequence identity for clustering (HPI used 0.95)")
    parser.add_argument("--coverage", type=float, default=0.9,
                        help="Minimum coverage (HPI used 0.9)")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def check_mmseqs2():
    """Check if MMseqs2 is installed."""
    if shutil.which("mmseqs") is None:
        print("ERROR: mmseqs2 not found in PATH.")
        print("Install: conda install -c bioconda mmseqs2")
        print("Or: https://github.com/soedinglab/MMseqs2")
        sys.exit(1)


def read_fasta(path: Path) -> dict:
    """Read FASTA file into {header: sequence} dict."""
    sequences = {}
    header = None
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    sequences[header] = "".join(seq_lines)
                header = line[1:].split()[0]  # Take first word as ID
                seq_lines = []
            elif line:
                seq_lines.append(line)
    if header is not None:
        sequences[header] = "".join(seq_lines)
    return sequences


def run_mmseqs2_clustering(input_fasta: Path, output_dir: Path,
                           min_seq_id: float, coverage: float, threads: int):
    """Run MMseqs2 easy-cluster."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "cluster"
    tmp_dir = output_dir / "tmp"

    cmd = [
        "mmseqs", "easy-cluster",
        str(input_fasta),
        str(prefix),
        str(tmp_dir),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", "1",        # coverage of shorter sequence
        "--cluster-mode", "0",    # greedy set cover
        "--threads", str(threads),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"MMseqs2 error:\n{result.stderr}")
        sys.exit(1)

    print(result.stdout)

    # Parse cluster TSV: cluster_rep \t member
    cluster_file = Path(str(prefix) + "_cluster.tsv")
    clusters = defaultdict(list)
    with open(cluster_file) as f:
        for line in f:
            rep, member = line.strip().split("\t")
            clusters[rep].append(member)

    rep_fasta = Path(str(prefix) + "_rep_seq.fasta")

    return clusters, rep_fasta


def split_clusters(clusters: dict, train_ratio: float, val_ratio: float,
                   test_ratio: float, seed: int):
    """Split cluster representatives into train/val/test sets."""
    reps = list(clusters.keys())
    random.seed(seed)
    random.shuffle(reps)

    n = len(reps)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_reps = set(reps[:n_train])
    val_reps = set(reps[n_train:n_train + n_val])
    test_reps = set(reps[n_train + n_val:])

    return train_reps, val_reps, test_reps


def write_split_fastas(sequences: dict, clusters: dict,
                       train_reps, val_reps, test_reps, output_dir: Path):
    """Write train/val/test FASTA files using cluster representatives."""
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_reps,
        "val": val_reps,
        "test": test_reps,
    }

    for split_name, reps in splits.items():
        out_path = output_dir / f"{split_name}.fasta"
        count = 0
        with open(out_path, "w") as f:
            for rep in reps:
                if rep in sequences:
                    seq = sequences[rep]
                    f.write(f">{rep}\n")
                    for i in range(0, len(seq), 80):
                        f.write(seq[i:i+80] + "\n")
                    count += 1
        print(f"{split_name}: {count} sequences -> {out_path}")


def main():
    args = parse_args()
    check_mmseqs2()

    print(f"=== MMseqs2 Deduplication & Splitting ===")
    print(f"Input: {args.input}")
    print(f"Min seq identity: {args.min_seq_id}")
    print(f"Coverage: {args.coverage}")
    print(f"Split: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")

    # Read all sequences
    sequences = read_fasta(args.input)
    print(f"Total input sequences: {len(sequences)}")

    # Cluster
    clusters, rep_fasta = run_mmseqs2_clustering(
        args.input, args.output_dir / "mmseqs2",
        args.min_seq_id, args.coverage, args.threads,
    )
    print(f"Clusters: {len(clusters)}")
    print(f"Cluster representatives: {len(clusters)}")
    print(f"Compression ratio: {len(sequences)}/{len(clusters)} = {len(sequences)/len(clusters):.1f}x")

    # Split by cluster
    train_reps, val_reps, test_reps = split_clusters(
        clusters, args.train_ratio, args.val_ratio, args.test_ratio, args.seed,
    )
    print(f"\nSplit: train={len(train_reps)}, val={len(val_reps)}, test={len(test_reps)}")

    # Write FASTA files
    write_split_fastas(sequences, clusters, train_reps, val_reps, test_reps, args.output_dir)

    print(f"\n=== Done. Output in {args.output_dir} ===")
    print("Next: update configs/preprocessing.yaml datapaths to point to train.fasta")


if __name__ == "__main__":
    main()
