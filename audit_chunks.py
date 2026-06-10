"""
Аудит качества чанков по всему корпусу.

Читает все data/processed/rag_chunks/*.rag_chunks.jsonl, считает:
- сколько чанков на парсер (docling / pymupdf / xlsx / vlm);
- разбивку по типам (text / table / visual);
- разбивку по source_group (тематической папке);
- статистику размеров (min/median/p90/max/avg + гистограмма);
- потенциальные проблемы: крошечные (<100 символов), огромные (>3000),
  без section_title;
- образцы самых маленьких и самых больших чанков.

Запускается после build_rag_chunks_from_records.py:
    python audit_chunks.py
    python audit_chunks.py --samples 5
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# HELPERS
# ============================================================

def get_parser_name(chunk: dict) -> str:
    """
    Имя парсера, который произвёл чанк. SemanticChunker кладёт его в
    metadata по-разному для текста (parser_names: список) и таблиц
    (parser_name: строка).
    """
    meta = chunk.get("metadata", {}) or {}
    parser_names = meta.get("parser_names")
    if parser_names and isinstance(parser_names, list):
        return parser_names[0]
    return meta.get("parser_name") or meta.get("parser_backend") or "unknown"


def truncate(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n] + "..."


# ============================================================
# REPORT
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Аудит rag_chunks по всему корпусу"
    )
    parser.add_argument(
        "--chunks-dir",
        default="data/processed/rag_chunks",
        help="Папка с *.rag_chunks.jsonl. Default: data/processed/rag_chunks",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Сколько крайних чанков показать (min/max). Default: 3",
    )
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.is_absolute():
        chunks_dir = PROJECT_ROOT / chunks_dir

    files = sorted(chunks_dir.glob("*.rag_chunks.jsonl"))
    if not files:
        print(f"Нет файлов *.rag_chunks.jsonl в {chunks_dir}")
        print("Сначала запусти: python build_rag_chunks_from_records.py")
        return

    all_chunks: list[dict] = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                all_chunks.append(json.loads(line))

    total = len(all_chunks)
    if total == 0:
        print(f"В файлах не нашлось ни одного чанка ({len(files)} файлов).")
        return

    print("=" * 78)
    print("CORPUS OVERVIEW")
    print("=" * 78)
    print(f"Файлов rag_chunks:  {len(files)}")
    print(f"Документов (uniq):  {len({c['source_file'] for c in all_chunks})}")
    print(f"Всего чанков:       {total}")

    # ---------------- BY PARSER ----------------
    by_parser: dict[str, list[dict]] = defaultdict(list)
    for c in all_chunks:
        by_parser[get_parser_name(c)].append(c)

    print()
    print("=" * 78)
    print("CHUNKS BY PARSER")
    print("=" * 78)
    print(f"{'parser':<26} {'chunks':>8} {'docs':>6} {'avg/doc':>9} {'median chars':>14}")
    for name, chunks in sorted(by_parser.items(), key=lambda x: -len(x[1])):
        docs = len({c["source_file"] for c in chunks})
        sizes = [len(c["text"]) for c in chunks]
        med = int(statistics.median(sizes)) if sizes else 0
        avg_per_doc = len(chunks) / max(1, docs)
        print(f"{name:<26} {len(chunks):>8} {docs:>6} {avg_per_doc:>9.1f} {med:>14}")

    # ---------------- BY TYPE ----------------
    by_type = Counter(c.get("chunk_type", "?") for c in all_chunks)
    print()
    print("=" * 78)
    print("CHUNKS BY TYPE")
    print("=" * 78)
    for t, n in by_type.most_common():
        print(f"  {t:<10} {n:>6}  ({100 * n / total:>5.1f}%)")

    # ---------------- BY SOURCE_GROUP ----------------
    by_group: dict[str, list[dict]] = defaultdict(list)
    for c in all_chunks:
        group = (c.get("metadata") or {}).get("source_group") or "unknown"
        by_group[group].append(c)

    print()
    print("=" * 78)
    print("CHUNKS BY SOURCE_GROUP")
    print("=" * 78)
    print(f"{'group':<30} {'chunks':>7} {'docs':>5} {'median':>7} {'avg':>6}")
    for group, chunks in sorted(by_group.items(), key=lambda x: -len(x[1])):
        docs = len({c["source_file"] for c in chunks})
        sizes = [len(c["text"]) for c in chunks]
        med = int(statistics.median(sizes)) if sizes else 0
        avg = int(sum(sizes) / max(1, len(sizes)))
        print(f"  {group:<28} {len(chunks):>7} {docs:>5} {med:>7} {avg:>6}")

    # ---------------- SIZE STATS ----------------
    sizes_sorted = sorted(len(c["text"]) for c in all_chunks)

    def pct(p: int) -> int:
        idx = min(len(sizes_sorted) - 1, int(len(sizes_sorted) * p / 100))
        return sizes_sorted[idx]

    print()
    print("=" * 78)
    print("CHUNK SIZE (chars)")
    print("=" * 78)
    print(f"  min:    {min(sizes_sorted)}")
    print(f"  p10:    {pct(10)}")
    print(f"  p25:    {pct(25)}")
    print(f"  median: {int(statistics.median(sizes_sorted))}")
    print(f"  p75:    {pct(75)}")
    print(f"  p90:    {pct(90)}")
    print(f"  max:    {max(sizes_sorted)}")
    print(f"  avg:    {int(sum(sizes_sorted) / len(sizes_sorted))}")

    # ---------------- HISTOGRAM ----------------
    buckets = [
        (0, 100, "<100"),
        (100, 300, "100-300"),
        (300, 800, "300-800"),
        (800, 1500, "800-1500"),
        (1500, 3000, "1500-3000"),
        (3000, 10 ** 9, ">=3000"),
    ]
    print()
    print("=" * 78)
    print("SIZE HISTOGRAM")
    print("=" * 78)
    for lo, hi, label in buckets:
        n = sum(1 for s in sizes_sorted if lo <= s < hi)
        share = 100 * n / total
        bar = "#" * int(share / 2)
        print(f"  {label:<12} {n:>5}  ({share:>5.1f}%)  {bar}")

    # ---------------- ISSUES ----------------
    tiny = [c for c in all_chunks if len(c["text"]) < 100]
    huge = [c for c in all_chunks if len(c["text"]) > 3000]
    no_section = [c for c in all_chunks if not c.get("section_title")]
    empty = [c for c in all_chunks if not (c.get("text") or "").strip()]

    print()
    print("=" * 78)
    print("POTENTIAL ISSUES")
    print("=" * 78)
    print(f"  Tiny chunks (<100 chars):  {len(tiny):>5}  ({100*len(tiny)/total:.1f}%)  → фрагментация")
    print(f"  Huge chunks (>3000 chars): {len(huge):>5}  ({100*len(huge)/total:.1f}%)  → могут проседать в embedding")
    print(f"  Без section_title:         {len(no_section):>5}  ({100*len(no_section)/total:.1f}%)  → потеря контекста раздела")
    print(f"  Пустых чанков:             {len(empty):>5}  (должно быть 0)")

    # ---------------- SAMPLES ----------------
    def show_samples(label: str, chunks: list[dict]) -> None:
        if not chunks:
            return
        n = min(args.samples, len(chunks))
        print()
        print("=" * 78)
        print(f"SAMPLE: {label}  (показано {n} из {len(chunks)})")
        print("=" * 78)
        for c in chunks[:n]:
            print(
                f"\n--- {c['source_file']} | type={c['chunk_type']} | "
                f"page={c.get('page_number')} | parser={get_parser_name(c)} | "
                f"size={len(c['text'])} ---"
            )
            print(truncate(c["text"], 500))

    smallest = sorted(all_chunks, key=lambda c: len(c["text"]))
    largest = sorted(all_chunks, key=lambda c: -len(c["text"]))

    show_samples("SMALLEST", smallest)
    show_samples("LARGEST", largest)


if __name__ == "__main__":
    main()
