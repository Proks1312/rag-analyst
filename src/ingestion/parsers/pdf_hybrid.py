from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ingestion.models import DocumentMetadata, ParsedDocument
from ingestion.parsers.base import BaseParser
from ingestion.parsers.pdf_docling import PdfDoclingParser
from ingestion.parsers.pdf_pymupdf import PdfPyMuPdfParser


class HybridDoclingPyMuPdfParser(BaseParser):
    """
    Hybrid PDF parser: пробует Docling, при падении или подозрительно
    плохом результате — переключается на PyMuPDF.

    Зачем нужен:
    - Docling даёт лучшую структуру (типизированные блоки, таблицы, page numbers),
      но падает на больших табличных PDF (например, std::bad_alloc на ВНИИПО 2024).
    - PyMuPDF дешёв и не падает, но даёт более грубое представление.

    Стратегия "hybrid_docling_then_pymupdf" из DocumentProcessingStrategy
    мапится именно на этот парсер.
    """

    parser_name: str = "hybrid_docling_pymupdf"
    parser_version: str = "0.1.0"
    supported_extensions: set[str] = {".pdf"}

    # Эвристики deg-output. Делаем мягко, чтобы не дёргать fallback на нормальных коротких PDF.
    min_blocks_threshold: int = 5
    min_avg_block_chars: int = 50

    def __init__(
        self,
        primary: PdfDoclingParser | None = None,
        fallback: PdfPyMuPdfParser | None = None,
        check_degraded_output: bool = True,
    ) -> None:
        self.primary = primary or PdfDoclingParser()
        self.fallback = fallback or PdfPyMuPdfParser()
        self.check_degraded_output = check_degraded_output

    def parse(
        self,
        path: Path,
        metadata: DocumentMetadata,
    ) -> ParsedDocument:
        # Попытка 1: Docling.
        try:
            document = self.primary.parse(
                path=path,
                metadata=self._fresh_metadata(metadata),
            )

            if self.check_degraded_output and self._looks_degraded(document):
                reason = self._degraded_reason(document)
                print(
                    f"[hybrid] Docling output looks degraded for {path.name}: {reason}. "
                    f"Falling back to PyMuPDF."
                )
                return self._fallback_parse(
                    path=path,
                    metadata=metadata,
                    failure_reason=f"degraded_output: {reason}",
                )

            self._mark_metadata_success(
                document=document,
                primary_used=True,
                failure_reason=None,
            )
            return document

        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            print(
                f"[hybrid] Docling failed on {path.name}: {failure_reason}. "
                f"Falling back to PyMuPDF."
            )
            return self._fallback_parse(
                path=path,
                metadata=metadata,
                failure_reason=failure_reason,
            )

    # ============================================================
    # FALLBACK
    # ============================================================

    def _fallback_parse(
        self,
        path: Path,
        metadata: DocumentMetadata,
        failure_reason: str,
    ) -> ParsedDocument:
        document = self.fallback.parse(
            path=path,
            metadata=self._fresh_metadata(metadata),
        )

        self._mark_metadata_success(
            document=document,
            primary_used=False,
            failure_reason=failure_reason,
        )

        return document

    # ============================================================
    # HEURISTICS
    # ============================================================

    def _looks_degraded(self, document: ParsedDocument) -> bool:
        blocks = document.non_empty_blocks()

        if len(blocks) < self.min_blocks_threshold and not document.non_empty_tables():
            return True

        if not blocks:
            return True

        total_chars = sum(len(b.block_text) for b in blocks)
        avg_chars = total_chars / max(1, len(blocks))

        if avg_chars < self.min_avg_block_chars and not document.non_empty_tables():
            return True

        return False

    def _degraded_reason(self, document: ParsedDocument) -> str:
        blocks = document.non_empty_blocks()
        tables = document.non_empty_tables()

        if not blocks:
            return "no blocks parsed"

        total_chars = sum(len(b.block_text) for b in blocks)
        avg_chars = total_chars / max(1, len(blocks))

        return (
            f"blocks={len(blocks)}, tables={len(tables)}, "
            f"avg_block_chars={avg_chars:.1f}"
        )

    # ============================================================
    # METADATA HELPERS
    # ============================================================

    @staticmethod
    def _fresh_metadata(metadata: DocumentMetadata) -> DocumentMetadata:
        """
        Делаем копию метаданных, чтобы primary и fallback не наследовали
        друг у друга parser_name / parser_version.
        """

        try:
            return replace(metadata)
        except TypeError:
            return metadata

    def _mark_metadata_success(
        self,
        document: ParsedDocument,
        primary_used: bool,
        failure_reason: str | None,
    ) -> None:
        meta = document.metadata

        # Внешнее имя парсера — hybrid, конкретный backend пишем в extra.
        meta.extra["hybrid_primary_used"] = primary_used
        meta.extra["hybrid_actual_backend"] = (
            self.primary.parser_name if primary_used else self.fallback.parser_name
        )

        if failure_reason:
            meta.extra["hybrid_primary_failure"] = failure_reason

        # parser_name/version оставляем тем, кто реально парсил —
        # это важно для downstream-логики (block_cleaner / table_extractor).
