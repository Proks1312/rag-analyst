"""
Сборка RAG-чанков из embedding_records JSONL.

В проекте уже есть скрипт build_semantic_rag_chunks.py — он работает с
data/processed/ingestion/*.blocks.jsonl + *.tables.jsonl, использует
домен-специфичные эвристики (пожары, "наименование показателя" и т.п.)
и был написан до того, как Docling-парсер начал выдавать настоящую
иерархию заголовков.

Этот скрипт — альтернативный entry-point поверх нового SemanticChunker
из ingestion.chunking.semantic_chunker. Он:

- читает data/processed/embedding_input/*.embedding_records.jsonl;
- использует section_path, который Docling-парсер кладёт в metadata
  каждого record-а (стек заголовков из iterate_items);
- группирует блоки одной секции в чанки нужного размера с overlap;
- таблицы кладёт отдельными чанками;
- пишет data/processed/rag_chunks/*.rag_chunks.jsonl.

Оба пути могут сосуществовать. Старый build_semantic_rag_chunks.py
полезен, когда мы хотим перебилдить чанки из уже сохранённых
blocks/tables без переэмбеддинга. Новый — когда работаем с
embedding_records, которые уже отсортированы и нормализованы
exporter'ом.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ============================================================
# PATH / IMPORT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from ingestion.chunking.semantic_chunker import SemanticChunker
from ingestion.exporters.embedding_exporter import EmbeddingRecord


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build RAG chunks from embedding_records JSONL "
            "using the new SemanticChunker (Docling section_path aware)."
        )
    )

    p.add_argument(
        "--input-dir",
        default="data/processed/embedding_input",
        help="Folder with *.embedding_records.jsonl files",
    )
    p.add_argument(
        "--output-dir",
        default="data/processed/rag_chunks",
        help="Folder to write *.rag_chunks.jsonl files into",
    )
    p.add_argument(
        "--input-file",
        default=None,
        help="(Optional) process only one *.embedding_records.jsonl",
    )
    p.add_argument("--target-chars", type=int, default=1500, help="Target chunk size in characters")
    p.add_argument("--max-chars", type=int, default=3000, help="Hard max chunk size in characters")
    p.add_argument("--min-chars", type=int, default=200, help="Min useful chunk size in characters")
    p.add_argument("--overlap-sentences", type=int, default=1, help="Sentences to overlap between chunks")
    p.add_argument(
        "--no-section-header",
        action="store_true",
        help="Do not prepend '[Раздел: ...]' header to chunk text",
    )

    return p.parse_args()


# ============================================================
# HELPERS
# ============================================================

def _resolve_dir(value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _find_input_files(input_dir: Path, single_file: Path | None) -> list[Path]:
    if single_file is not None:
        if not single_file.is_absolute():
            single_file = PROJECT_ROOT / single_file
        if not single_file.exists():
            raise FileNotFoundError(f"Input file not found: {single_file}")
        return [single_file]

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    files = sorted(input_dir.glob("*.embedding_records.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No *.embedding_records.jsonl files found in {input_dir}"
        )

    return files


def _output_path_for(input_path: Path, output_dir: Path) -> Path:
    name = input_path.name.replace(".embedding_records.jsonl", ".rag_chunks.jsonl")
    return output_dir / name


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    input_dir = _resolve_dir(args.input_dir)
    output_dir = _resolve_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    single_file = Path(args.input_file) if args.input_file else None
    input_files = _find_input_files(input_dir=input_dir, single_file=single_file)

    chunker = SemanticChunker(
        target_chunk_chars=args.target_chars,
        max_chunk_chars=args.max_chars,
        min_chunk_chars=args.min_chars,
        overlap_sentences=args.overlap_sentences,
        prepend_section_header=not args.no_section_header,
    )

    print("=" * 80)
    print("BUILD RAG CHUNKS FROM EMBEDDING RECORDS")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input dir:    {input_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Files:        {len(input_files)}")
    print(f"target_chars: {args.target_chars}")
    print(f"max_chars:    {args.max_chars}")
    print(f"min_chars:    {args.min_chars}")
    print(f"overlap_sent: {args.overlap_sentences}")
    print(f"section hdr:  {not args.no_section_header}")
    print("=" * 80)

    total_records = 0
    total_chunks = 0

    for input_path in input_files:
        records: list[EmbeddingRecord] = SemanticChunker.load_records_jsonl(input_path)
        chunks = chunker.chunk_records(records)
        output_path = _output_path_for(input_path, output_dir)
        chunker.save_jsonl(chunks=chunks, output_path=output_path)

        total_records += len(records)
        total_chunks += len(chunks)

        avg_chars = (
            round(sum(len(c.text) for c in chunks) / len(chunks))
            if chunks
            else 0
        )

        print(
            f"  {input_path.name:60s} → "
            f"{len(records):4d} records → {len(chunks):4d} chunks "
            f"(avg {avg_chars} chars) → {output_path.name}"
        )

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Total records: {total_records}")
    print(f"Total chunks:  {total_chunks}")
    print("=" * 80)


if __name__ == "__main__":
    main()
