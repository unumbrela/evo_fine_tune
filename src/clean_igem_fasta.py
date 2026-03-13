"""
清洗 iGEM FASTA 数据，使其适合 Evo2 微调。

过滤规则 (参考 HPI-Potsdam):
  1. 最小长度 100bp (太短的序列无法提供有效上下文)
  2. 去除低复杂度序列 (如 poly-T, poly-A 等)
  3. 仅保留 ATCG 字符
  4. 序列去重 (基于 SHA256)
  5. 可选: 仅保留 composite 类型 (HPI 发现这对微调效果最好)

用法:
    python src/clean_igem_fasta.py \
        --input data/raw/igem_sequences.fasta \
        --output data/raw/igem_cleaned.fasta \
        --min-length 100 \
        --composite-only
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Clean iGEM FASTA for Evo2 fine-tuning")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-length", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=50000)
    parser.add_argument("--composite-only", action="store_true",
                        help="Only keep composite parts")
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--min-complexity", type=float, default=0.3,
                        help="Minimum sequence complexity (unique 3-mers / possible 3-mers)")
    return parser.parse_args()


def read_fasta(path: Path):
    """Yield (header_line, sequence) tuples."""
    header = None
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line
                seq_lines = []
            elif line:
                seq_lines.append(line.upper())
    if header is not None:
        yield header, "".join(seq_lines)


def sequence_complexity(seq: str, k: int = 3) -> float:
    """
    Sequence complexity as ratio of unique k-mers to total k-mers.
    Low-complexity sequences (poly-T, poly-A) have very low values.
    """
    if len(seq) < k:
        return 0.0
    kmers = set()
    for i in range(len(seq) - k + 1):
        kmers.add(seq[i:i+k])
    possible = min(4**k, len(seq) - k + 1)
    return len(kmers) / possible


def get_part_type(header: str) -> str:
    """Extract part type from header line."""
    match = re.search(r"type=(\S+)", header)
    return match.group(1) if match else "unknown"


def main():
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    reasons = {
        "short": 0,
        "long": 0,
        "non_atcg": 0,
        "low_complexity": 0,
        "duplicate": 0,
        "not_composite": 0,
    }
    seen_hashes = set()

    with open(args.output, "w") as out:
        for header, seq in read_fasta(args.input):
            total += 1

            # Length filter
            if len(seq) < args.min_length:
                reasons["short"] += 1
                continue

            if len(seq) > args.max_length:
                reasons["long"] += 1
                continue

            # ATCG-only
            if not re.match(r"^[ATCG]+$", seq):
                reasons["non_atcg"] += 1
                continue

            # Complexity filter
            if sequence_complexity(seq) < args.min_complexity:
                reasons["low_complexity"] += 1
                continue

            # Composite-only filter
            if args.composite_only:
                ptype = get_part_type(header)
                if ptype != "composite":
                    reasons["not_composite"] += 1
                    continue

            # Dedup
            if not args.no_dedup:
                h = hashlib.sha256(seq.encode()).hexdigest()[:16]
                if h in seen_hashes:
                    reasons["duplicate"] += 1
                    continue
                seen_hashes.add(h)

            # Write
            out.write(f"{header}\n")
            for i in range(0, len(seq), 80):
                out.write(seq[i:i+80] + "\n")
            kept += 1

    print(f"=== Cleaning Results ===")
    print(f"Input:  {total} sequences")
    print(f"Output: {kept} sequences")
    print(f"Removed:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {reason}: {count}")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
