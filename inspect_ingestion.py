"""
Диагностика: смотрим, что получилось у парсера на одном документе и
какие чанки из этого соберёт semantic chunker — БЕЗ повторного прогона
парсинга.

Скрипт читает уже сохранённый набор data/processed/ingestion/<stem>.{blocks,tables,visuals}.jsonl,
восстанавливает ParsedDocument, прогоняет через EmbeddingExporter и
SemanticChunker и печатает сводку + образцы.

Пример:
    python inspect_ingestion.py --stem "Электрокабель (2026, report)"
    python inspect_ingestion.py --stem "АО_СИБКАБЕЛЬ-2026-04-16" --samples 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingestion.models import (
    DocumentMetadata,
    ParsedBlock,
    ParsedDocument,
    ParsedTable,
    ParsedVisual,
)
from ingestion.exporters.embedding_exporter import EmbeddingExporter
from ingestion.chunking.semantic_chunker import SemanticChunker


# ============================================================
# LOAD STORAGE JSONL -> ParsedDocument
# ============================================================

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_parsed_document(stem: str, ingest_dir: Path) -> ParsedDocument:
    blocks_rows = _read_jsonl(ingest_dir / f"{stem}.blocks.jsonl")
    tables_rows = _read_jsonl(ingest_dir / f"{stem}.tables.jsonl")
    visuals_rows = _read_jsonl(ingest_dir / f"{stem}.visuals.jsonl")

    all_rows = blocks_rows + tables_rows + visuals_rows
    if not all_rows:
        raise FileNotFoundError(
            f"Не нашёл ни одного *.jsonl для stem={stem!r} в {ingest_dir}. "
            f"Сначала прогони ingestion: run_ingestion_file.py --pdf ..."
        )

    metadata_dict: Optional[dict] = None
    for row in all_rows:
        if "document" in row:
            metadata_dict = row["document"]
            break

    if metadata_dict is None:
        raise RuntimeError("В jsonl нет metadata документа (поле 'document').")

    metadata = DocumentMetadata(**metadata_dict)

    blocks = [ParsedBlock(**row["entity"]) for row in blocks_rows]
    tables = [ParsedTable(**row["entity"]) for row in tables_rows]
    visuals = [ParsedVisual(**row["entity"]) for row in visuals_rows]

    return ParsedDocument(
        metadata=metadata,
        blocks=blocks,
        tables=tables,
        visuals=visuals,
    )


# ============================================================
# REPORT
# ============================================================

def _truncate(text: str, n: int) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    return text[:n] + "..."


def report(doc: ParsedDocument, samples: int) -> None:
    blocks = doc.non_empty_blocks()
    tables = doc.non_empty_tables()
    visuals = doc.non_empty_visuals()

    print("=" * 80)
    print("DOCUMENT")
    print("=" * 80)
    print(f"source_file:     {doc.metadata.source_file}")
    print(f"parser_name:     {doc.metadata.parser_name}")
    print(f"parser_version:  {doc.metadata.parser_version}")
    print(f"language:        {doc.metadata.language}")
    print(f"source_group:    {doc.metadata.extra.get('source_group')}")
    print(f"selected_strategy: {doc.metadata.extra.get('selected_strategy')}")
    print(f"blocks:          {len(blocks)}")
    print(f"tables:          {len(tables)}")
    print(f"visuals:         {len(visuals)}")

    btypes = Counter(b.block_type for b in blocks)
    print(f"block types:     {dict(btypes)}")

    # Embedding records (как делает exporter)
    exporter = EmbeddingExporter()
    records = exporter.export_records(doc)
    print(f"embedding records: {len(records)}")

    # Semantic chunks
    chunker = SemanticChunker()
    chunks = chunker.chunk_records(records)

    print()
    print("=" * 80)
    print("CHUNKS")
    print("=" * 80)
    print(f"total chunks:    {len(chunks)}")
    if chunks:
        ctypes = Counter(c.chunk_type for c in chunks)
        sizes = [len(c.text) for c in chunks]
        print(f"chunk types:     {dict(ctypes)}")
        print(
            f"chunk size chars: min={min(sizes)} avg={sum(sizes)//len(sizes)} "
            f"max={max(sizes)}"
        )

    # SAMPLES
    print()
    print("=" * 80)
    print(f"SAMPLE BLOCKS (first {min(samples, len(blocks))})")
    print("=" * 80)
    for i, b in enumerate(blocks[:samples], 1):
        print(
            f"\n--- block {i}: type={b.block_type}  page={b.page_number}  "
            f"section={b.section_title!r} ---"
        )
        print(_truncate(b.block_text, 500))

    print()
    print("=" * 80)
    print(f"SAMPLE TABLES (first {min(samples, len(tables))})")
    print("=" * 80)
    for i, t in enumerate(tables[:samples], 1):
        print(
            f"\n--- table {i}: page={t.page_number}  title={t.table_title!r}  "
            f"quality={t.metadata.get('table_quality')} ---"
        )
        print(_truncate(t.markdown_table, 700))

    print()
    print("=" * 80)
    print(f"SAMPLE CHUNKS (first {min(samples, len(chunks))})")
    print("=" * 80)
    for i, c in enumerate(chunks[:samples], 1):
        print(
            f"\n--- chunk {i}: type={c.chunk_type}  page={c.page_number}  "
            f"section={c.section_title!r}  len={len(c.text)} ---"
        )
        print(_truncate(c.text, 700))


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnostic: show what an ingestion run produced and "
                    "what semantic chunks come out of it."
    )
    parser.add_argument(
        "--stem",
        required=True,
        help="Имя документа без расширения, напр. 'Электрокабель (2026, report)'",
    )
    parser.add_argument(
        "--ingest-dir",
        default="data/processed/ingestion",
        help="Папка с *.blocks/tables/visuals.jsonl",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Сколько образцов блоков / таблиц / чанков печатать. Default: 3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ingest_dir = Path(args.ingest_dir)
    if not ingest_dir.is_absolute():
        ingest_dir = PROJECT_ROOT / ingest_dir

    doc = load_parsed_document(stem=args.stem, ingest_dir=ingest_dir)
    report(doc, samples=args.samples)


if __name__ == "__main__":
    main()
