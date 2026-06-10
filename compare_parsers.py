"""
Прогнать один PDF через несколько парсеров и сравнить результат.

Запускает Docling, PyMuPDF, Hybrid (Docling → PyMuPDF) на одном файле,
печатает сводку: число блоков/таблиц/символов, размер таблиц,
наличие section_path. По желанию выводит markdown N-ной таблицы,
чтобы глазами оценить, кто как разобрал многострочные заголовки и
группировки внутри.

VLM-парсер по умолчанию не запускается (слишком долго) — добавь --with-vlm,
если хочешь и его сравнить.

Использование:
    .\\.venv\\Scripts\\python.exe compare_parsers.py ^
        --pdf "data/raw/market reports (rus, cable)/ВНИИПО (2024).pdf" ^
        --output-dir data/processed/_compare_vniipo ^
        --show-table 9

    # Только Docling и PyMuPDF, без Hybrid:
    .\\.venv\\Scripts\\python.exe compare_parsers.py --pdf ... --parsers docling,pymupdf

    # Добавить VLM (qwen3-vl:8b через Ollama):
    .\\.venv\\Scripts\\python.exe compare_parsers.py --pdf ... --with-vlm
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from ingestion.models import DocumentMetadata, ParsedDocument


# ============================================================
# PARSER FACTORIES
# ============================================================

def _build_parser(name: str):
    """Возвращает инстанс парсера по короткому имени."""

    if name == "docling":
        from ingestion.parsers.pdf_docling import PdfDoclingParser
        return PdfDoclingParser()

    if name == "pymupdf":
        from ingestion.parsers.pdf_pymupdf import PdfPyMuPdfParser
        return PdfPyMuPdfParser()

    if name == "hybrid":
        from ingestion.parsers.pdf_docling import PdfDoclingParser
        from ingestion.parsers.pdf_pymupdf import PdfPyMuPdfParser
        from ingestion.parsers.pdf_hybrid import HybridDoclingPyMuPdfParser
        return HybridDoclingPyMuPdfParser(
            primary=PdfDoclingParser(),
            fallback=PdfPyMuPdfParser(),
        )

    if name == "vlm":
        from ingestion.parsers.pdf_vlm import VlmPdfParser
        return VlmPdfParser()

    raise ValueError(f"Unknown parser: {name}")


# ============================================================
# RUN ONE PARSER
# ============================================================

def run_parser(name: str, pdf_path: Path, out_root: Path) -> dict:
    """
    Прогоняет один парсер на pdf_path, сохраняет blocks/tables в JSONL,
    возвращает summary dict.
    """

    print(f"\n{'=' * 72}")
    print(f"[{name.upper()}] starting...")
    print("=" * 72)

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = DocumentMetadata(
        source_file=pdf_path.name,
        source_path=str(pdf_path),
        source_type="pdf",
        language="ru",
    )

    summary: dict = {
        "parser": name,
        "ok": False,
        "elapsed_s": 0.0,
        "blocks": 0,
        "tables": 0,
        "total_block_chars": 0,
        "total_table_chars": 0,
        "avg_block_chars": 0,
        "max_block_chars": 0,
        "blocks_with_section": 0,
        "tables_with_section": 0,
        "tables_with_title": 0,
        "avg_table_chars": 0,
        "max_table_chars": 0,
        "table_rows_total": 0,
        "error": None,
        "extra": {},
    }

    t0 = time.time()
    try:
        parser = _build_parser(name)
        doc: ParsedDocument = parser.parse(pdf_path, metadata)
    except Exception as exc:
        summary["elapsed_s"] = round(time.time() - t0, 2)
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[{name}] FAILED: {summary['error']}")
        traceback.print_exc()
        return summary

    summary["elapsed_s"] = round(time.time() - t0, 2)
    summary["ok"] = True
    summary["extra"] = {
        "docling_first_error": metadata.extra.get("docling_first_error"),
        "docling_used_repaired_pdf": metadata.extra.get("docling_used_repaired_pdf"),
        "hybrid_primary_used": metadata.extra.get("hybrid_primary_used"),
        "fallback_parser": metadata.extra.get("fallback_parser"),
        "pdf_repaired": metadata.extra.get("pdf_repaired"),
    }

    blocks = [b for b in doc.blocks if b.block_text and b.block_text.strip()]
    tables = [t for t in doc.tables if t.markdown_table and t.markdown_table.strip()]

    summary["blocks"] = len(blocks)
    summary["tables"] = len(tables)

    if blocks:
        sizes = [len(b.block_text) for b in blocks]
        summary["total_block_chars"] = sum(sizes)
        summary["avg_block_chars"] = int(statistics.mean(sizes))
        summary["max_block_chars"] = max(sizes)
        summary["blocks_with_section"] = sum(1 for b in blocks if b.section_title)

    if tables:
        tsizes = [len(t.markdown_table) for t in tables]
        summary["total_table_chars"] = sum(tsizes)
        summary["avg_table_chars"] = int(statistics.mean(tsizes))
        summary["max_table_chars"] = max(tsizes)
        summary["tables_with_section"] = sum(
            1 for t in tables if t.metadata.get("section_path")
        )
        summary["tables_with_title"] = sum(1 for t in tables if t.table_title)
        summary["table_rows_total"] = sum(len(t.raw_data) for t in tables)

    # Save blocks/tables for inspection.
    blocks_path = out_dir / "blocks.jsonl"
    with blocks_path.open("w", encoding="utf-8") as f:
        for b in blocks:
            f.write(json.dumps(asdict(b), ensure_ascii=False) + "\n")

    tables_path = out_dir / "tables.jsonl"
    with tables_path.open("w", encoding="utf-8") as f:
        for t in tables:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")

    print(
        f"[{name}] ok in {summary['elapsed_s']}s: "
        f"blocks={summary['blocks']}, tables={summary['tables']}"
    )
    return summary


# ============================================================
# PRINT REPORT
# ============================================================

def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def print_report(summaries: list[dict], show_table: int | None) -> None:
    print()
    print("=" * 88)
    print("СВОДКА")
    print("=" * 88)

    rows = [
        ("parser",            "parser"),
        ("ok",                "ok"),
        ("elapsed_s",         "time, s"),
        ("blocks",            "blocks"),
        ("avg_block_chars",   "avg block, ch"),
        ("max_block_chars",   "max block, ch"),
        ("blocks_with_section", "blocks w/section"),
        ("tables",            "tables"),
        ("avg_table_chars",   "avg table, ch"),
        ("max_table_chars",   "max table, ch"),
        ("table_rows_total",  "Σ table rows"),
        ("tables_with_title", "tables w/title"),
        ("tables_with_section", "tables w/section"),
        ("error",             "error"),
    ]

    parsers = [s["parser"] for s in summaries]
    col_w = max(20, max(len(p) for p in parsers) + 2)

    header = f"{'metric':<22}" + "".join(f"{p:>{col_w}}" for p in parsers)
    print(header)
    print("-" * len(header))
    for key, label in rows:
        line = f"{label:<22}" + "".join(
            f"{_fmt(s.get(key)):>{col_w}}" for s in summaries
        )
        print(line)

    # Extra flags (Docling repair etc.)
    print()
    print("EXTRA FLAGS")
    print("-" * 88)
    for s in summaries:
        if s.get("extra"):
            non_null = {k: v for k, v in s["extra"].items() if v}
            if non_null:
                print(f"  [{s['parser']}] {non_null}")

    # Sample table.
    if show_table is not None:
        print()
        print("=" * 88)
        print(f"ОБРАЗЕЦ ТАБЛИЦЫ #{show_table} (1-indexed)")
        print("=" * 88)
        for s in summaries:
            if not s["ok"]:
                continue
            parser = s["parser"]
            tables_path = Path(s["out_dir"]) / "tables.jsonl"
            tables = [json.loads(l) for l in tables_path.read_text(
                encoding="utf-8"
            ).splitlines() if l.strip()]
            if not tables:
                print(f"\n--- [{parser}] нет таблиц ---")
                continue
            idx = max(0, min(show_table - 1, len(tables) - 1))
            t = tables[idx]
            print(f"\n--- [{parser}] table #{idx + 1} (page={t.get('page_number')}) ---")
            print(f"title:        {t.get('table_title')}")
            print(f"section_path: {(t.get('metadata') or {}).get('section_path')}")
            md = t.get("markdown_table", "") or ""
            if len(md) > 2000:
                md = md[:2000] + f"\n... (+{len(md) - 2000} символов)"
            print(md)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare PDF parsers on one document."
    )
    ap.add_argument("--pdf", required=True)
    ap.add_argument(
        "--output-dir",
        default="data/processed/_compare",
        help="Куда складывать blocks.jsonl / tables.jsonl для каждого парсера.",
    )
    ap.add_argument(
        "--parsers",
        default="docling,pymupdf,hybrid",
        help="Через запятую: docling, pymupdf, hybrid (vlm — отдельно через --with-vlm).",
    )
    ap.add_argument(
        "--with-vlm",
        action="store_true",
        help="Добавить VLM-парсер (qwen3-vl:8b через Ollama). Долго.",
    )
    ap.add_argument(
        "--show-table",
        type=int,
        default=None,
        help="Показать markdown N-ной таблицы (1-indexed) для каждого парсера.",
    )
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    requested = [p.strip() for p in args.parsers.split(",") if p.strip()]
    if args.with_vlm and "vlm" not in requested:
        requested.append("vlm")

    print("=" * 88)
    print("COMPARE PARSERS")
    print("=" * 88)
    print(f"PDF:     {pdf_path}")
    print(f"Out:     {out_root}")
    print(f"Parsers: {requested}")
    print("=" * 88)

    summaries: list[dict] = []
    for name in requested:
        s = run_parser(name, pdf_path, out_root)
        s["out_dir"] = str(out_root / name)
        summaries.append(s)

    print_report(summaries, show_table=args.show_table)

    # Save aggregate summary.
    summary_path = out_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
