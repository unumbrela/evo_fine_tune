"""
iGEM Registry 数据预处理脚本

将 iGEM Registry XML dump 转换为可用于 Evo2 微调的 FASTA 文件。
参考 HPI-Potsdam 的数据预处理流程:
  - 去除 deleted/red-list parts
  - 过滤短序列 (< 5bp)
  - 仅保留 ATCG 字符
  - 去重 (基于序列 hash)
  - 可选: 仅保留 composite parts (HPI 发现这对微调效果最好)

用法:
    python src/prepare_igem_data.py \\
        --xml-dump data/raw/igem_dump.xml \\
        --output data/raw/igem_sequences.fasta \\
        --min-length 100 \\
        --composite-only

数据获取:
    iGEM Registry XML dump 可从 iGEM 官方获取
    或使用 iGEM API: https://parts.igem.org/cgi/xml/part.cgi?part=BBa_XXXX
"""

import argparse
import hashlib
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess iGEM Registry data for Evo2 fine-tuning")
    parser.add_argument("--xml-dump", type=Path, required=True, help="Path to iGEM XML dump file")
    parser.add_argument("--output", type=Path, default=Path("data/raw/igem_sequences.fasta"))
    parser.add_argument("--min-length", type=int, default=100, help="Minimum sequence length (bp)")
    parser.add_argument("--max-length", type=int, default=50000, help="Maximum sequence length (bp)")
    parser.add_argument("--composite-only", action="store_true",
                        help="Only keep composite parts (recommended by HPI-Potsdam)")
    parser.add_argument("--no-dedup", action="store_true", help="Skip deduplication")
    return parser.parse_args()


def clean_sequence(seq: str) -> str | None:
    """Clean and validate a DNA sequence. Returns None if invalid."""
    seq = seq.upper().strip()
    seq = re.sub(r"\s+", "", seq)

    # Only keep pure ATCG sequences
    if not re.match(r"^[ATCG]+$", seq):
        return None

    return seq


def seq_hash(seq: str) -> str:
    """SHA256 hash of sequence for deduplication."""
    return hashlib.sha256(seq.encode()).hexdigest()[:16]


def parse_igem_xml(xml_path: Path, min_length: int, max_length: int, composite_only: bool):
    """Parse iGEM XML dump and yield (part_name, sequence, part_type) tuples."""
    print(f"Parsing XML: {xml_path}")

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"XML parse error: {e}")
        print("Trying with encoding fix...")
        with open(xml_path, "rb") as f:
            content = f.read()
        # Remove invalid XML bytes
        content = re.sub(rb"[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]", b"", content)
        root = ET.fromstring(content)

    # iGEM XML structure varies; adapt to actual format
    parts_found = 0
    parts_kept = 0

    for part in root.iter("part"):
        parts_found += 1

        part_name = part.findtext("part_name", "")
        sequence = part.findtext("sequences/seq_data", "")
        if not sequence:
            sequence = part.findtext("sequence", "")
        part_type = part.findtext("part_type", "").lower()
        status = part.findtext("status", "").lower()

        # Skip deleted / unavailable parts
        if "deleted" in status or "fail" in status:
            continue

        # Composite-only filter
        if composite_only and "composite" not in part_type:
            continue

        # Clean sequence
        cleaned = clean_sequence(sequence)
        if cleaned is None:
            continue

        # Length filter
        if len(cleaned) < min_length or len(cleaned) > max_length:
            continue

        parts_kept += 1
        yield part_name, cleaned, part_type

    print(f"Parsed {parts_found} parts, kept {parts_kept}")


def write_fasta(records, output_path: Path, deduplicate: bool = True):
    """Write records to FASTA, optionally deduplicating by sequence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    written = 0
    duplicates = 0

    with open(output_path, "w") as f:
        for name, seq, part_type in records:
            if deduplicate:
                h = seq_hash(seq)
                if h in seen_hashes:
                    duplicates += 1
                    continue
                seen_hashes.add(h)

            f.write(f">{name} type={part_type} len={len(seq)}\n")
            # Write sequence in 80-char lines
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")
            written += 1

    print(f"Written {written} sequences to {output_path}")
    if deduplicate:
        print(f"Removed {duplicates} duplicate sequences")
    return written


def main():
    args = parse_args()

    if not args.xml_dump.exists():
        print(f"ERROR: XML dump not found: {args.xml_dump}")
        print("")
        print("iGEM Registry XML dump 获取方式:")
        print("  1. 联系 iGEM 获取完整 dump")
        print("  2. 使用 API 逐个下载: https://parts.igem.org/cgi/xml/part.cgi?part=BBa_XXXX")
        print("  3. 使用 BioComplete API: https://biocomplete.it/")
        sys.exit(1)

    records = parse_igem_xml(
        args.xml_dump,
        min_length=args.min_length,
        max_length=args.max_length,
        composite_only=args.composite_only,
    )

    write_fasta(
        records,
        args.output,
        deduplicate=not args.no_dedup,
    )


if __name__ == "__main__":
    main()
