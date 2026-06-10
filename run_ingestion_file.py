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


from ingestion.pipeline import DocumentIngestionPipeline
from ingestion.registry import ParserRegistry
from ingestion.parsers.pdf_docling import PdfDoclingParser
from ingestion.parsers.pdf_pymupdf import PdfPyMuPdfParser
from ingestion.parsers.pdf_hybrid import HybridDoclingPyMuPdfParser
from ingestion.parsers.xlsx_parser import XlsxParser
from ingestion.parsers.pdf_vlm import VlmPdfParser
from ingestion.storage.local_storage import LocalJsonlStorage
from ingestion.profiling.document_profiler import DocumentProfiler
from ingestion.strategies.strategy_router import StrategyRouter


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-stage modular ingestion for one document"
    )

    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--project", default="RAG Analyst", help="Project metadata")
    parser.add_argument("--market", default=None, help="Market metadata")
    parser.add_argument("--product", default=None, help="Product metadata")

    parser.add_argument(
        "--language",
        default="unknown",
        choices=["ru", "en", "zh", "mixed", "unknown"],
        help="Document language metadata",
    )

    parser.add_argument(
        "--source-group",
        default="general",
        help="Source group metadata, e.g. market_reports_cable, market_reports_ath, financial_pdf",
    )

    parser.add_argument(
        "--source-type",
        default="pdf",
        help="Source type from manifest. Stored as manifest_source_type.",
    )

    parser.add_argument(
        "--topic",
        default=None,
        help="Topic metadata, e.g. cable_market, ath_mdh, counterparty_context",
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed/ingestion",
        help="Output folder for blocks/tables/visuals jsonl",
    )

    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Do not render PDF pages to PNG",
    )

    parser.add_argument(
        "--no-embedding-records",
        action="store_true",
        help="Do not export embedding records JSONL",
    )

    parser.add_argument(
        "--parser",
        default="auto",
        choices=[
            "auto",
            "profile",
            "docling",
            "pymupdf",
            "hybrid",
            "vlm",
            "docling_full",
            "pymupdf_text",
            "pymupdf_table_like",
            "hybrid_docling_then_pymupdf",
            "ocr_required",
        ],
        help=(
            "Parser / strategy mode. Default: auto — профилировщик сам "
            "выбирает стратегию для каждого документа. При падении Docling "
            "pipeline переключается на PyMuPDF."
        ),
    )

    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Only profile document and exit",
    )

    return parser.parse_args()


# ============================================================
# REGISTRY + PIPELINE BUILDER
# ============================================================

def build_registry() -> ParserRegistry:
    registry = ParserRegistry()

    docling = PdfDoclingParser()
    registry.register(docling)

    pymupdf = PdfPyMuPdfParser()
    registry.register(pymupdf)

    hybrid = HybridDoclingPyMuPdfParser(primary=docling, fallback=pymupdf)
    registry.register(hybrid)

    # Excel-книги (.xlsx / .xlsm). По расширению не пересекается с PDF-парсерами.
    registry.register(XlsxParser())

    # VLM-парсер на базе Qwen3-VL через Ollama. Реализует слот стратегии
    # ocr_required (через _resolve_parser_for_strategy). Вызывается явно
    # через --parser vlm или --parser ocr_required, либо профилировщиком
    # для сканированных документов.
    registry.register(VlmPdfParser())

    return registry


def build_pipeline(
    output_dir: Path,
    render_visuals: bool,
    export_embedding_records: bool,
) -> DocumentIngestionPipeline:
    registry = build_registry()
    storage = LocalJsonlStorage(output_dir=output_dir)

    return DocumentIngestionPipeline(
        parser_registry=registry,
        storage=storage,
        profiler=DocumentProfiler(),
        router=StrategyRouter(),
        render_visuals=render_visuals,
        export_embedding_records=export_embedding_records,
    )


# ============================================================
# STRATEGY RESOLUTION
# ============================================================

_ALIAS_TO_STRATEGY: dict[str, str] = {
    "docling": "docling_full",
    "pymupdf": "pymupdf_text",
    "hybrid": "hybrid_docling_then_pymupdf",
    "vlm": "ocr_required",  # VLM-парсер занимает слот ocr_required
}

_DIRECT_STRATEGIES: set[str] = {
    "docling_full",
    "pymupdf_text",
    "pymupdf_table_like",
    "hybrid_docling_then_pymupdf",
    "ocr_required",
}


def resolve_strategy_override(parser_arg: str) -> str | None:
    if parser_arg == "auto":
        return None

    if parser_arg == "profile":
        return None

    if parser_arg in _ALIAS_TO_STRATEGY:
        return _ALIAS_TO_STRATEGY[parser_arg]

    if parser_arg in _DIRECT_STRATEGIES:
        return parser_arg

    raise ValueError(f"Unknown --parser value: {parser_arg}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    render_visuals = not args.no_visuals
    export_embedding_records = not args.no_embedding_records

    print("=" * 80)
    print("RUN MULTI-STAGE INGESTION FILE")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"PDF: {pdf_path}")
    print(f"Output dir: {output_dir}")
    print(f"Project: {args.project}")
    print(f"Market: {args.market}")
    print(f"Product: {args.product}")
    print(f"Language: {args.language}")
    print(f"Source group: {args.source_group}")
    print(f"Manifest source type: {args.source_type}")
    print(f"Topic: {args.topic}")
    print(f"Render visuals: {render_visuals}")
    print(f"Export embedding records: {export_embedding_records}")
    print(f"Parser/strategy mode: {args.parser}")
    print("=" * 80)

    if args.profile_only or args.parser == "profile":
        profiler = DocumentProfiler()
        router = StrategyRouter()
        profile = profiler.profile(pdf_path)
        print(router.explain(profile))
        print("Profile only. Exit.")
        return

    pipeline = build_pipeline(
        output_dir=output_dir,
        render_visuals=render_visuals,
        export_embedding_records=export_embedding_records,
    )

    strategy_override = resolve_strategy_override(args.parser)

    document = pipeline.run(
        path=pdf_path,
        project=args.project,
        market=args.market,
        product=args.product,
        language=args.language,
        source_group=args.source_group,
        manifest_source_type=args.source_type,
        topic=args.topic,
        strategy_override=strategy_override,
    )

    print("=" * 80)
    print("INGESTION FILE DONE")
    print("=" * 80)
    print(f"Source file: {document.metadata.source_file}")
    print(f"Parser: {document.metadata.parser_name}")
    print(f"Selected strategy: {document.metadata.extra.get('selected_strategy')}")
    print(f"Parser backend: {document.metadata.extra.get('parser_backend')}")
    print(f"Fallback parser: {document.metadata.extra.get('fallback_parser')}")
    print(f"PDF repaired: {document.metadata.extra.get('pdf_repaired')}")
    print(f"Source group: {document.metadata.extra.get('source_group')}")
    print(f"Manifest source type: {document.metadata.extra.get('manifest_source_type')}")
    print(f"Topic: {document.metadata.extra.get('topic')}")
    print(f"Hybrid primary used: {document.metadata.extra.get('hybrid_primary_used')}")
    print(f"Blocks:  {len(document.non_empty_blocks())}")
    print(f"Tables:  {len(document.non_empty_tables())}")
    print(f"Visuals: {len(document.non_empty_visuals())}")
    print("=" * 80)


if __name__ == "__main__":
    main()