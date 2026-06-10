from __future__ import annotations

from pathlib import Path

from ingestion.cleaners.block_cleaner import BlockCleaner
from ingestion.cleaners.table_extractor import TableExtractor
from ingestion.cleaners.table_normalizer import TableNormalizer
from ingestion.exporters.embedding_exporter import EmbeddingExporter
from ingestion.models import DocumentMetadata, ParsedDocument, SourceType, Language
from ingestion.parsers.base import BaseParser
from ingestion.profiling.document_profiler import DocumentProfiler
from ingestion.profiling.profile_models import DocumentProcessingStrategy
from ingestion.registry import ParserRegistry
from ingestion.storage.local_storage import LocalJsonlStorage
from ingestion.strategies.strategy_router import StrategyRouter
from ingestion.visual.pdf_renderer import PdfPageRenderer


class DocumentIngestionPipeline:
    """
    Multi-stage document processing pipeline.

    Stage 1: source metadata
    Stage 2: document profile
    Stage 3: strategy routing
    Stage 4: parser execution
    Stage 4.1: if Docling fails -> fallback to PyMuPDF
    Stage 5: block cleaning
    Stage 6: table extraction
    Stage 7: table normalization
    Stage 8: optional visual rendering
    Stage 9: local storage
    Stage 10: embedding records export

    Про repair PDF:
    Чинить «битые» PDF (пересохранение через PyMuPDF) умеет сам
    PdfDoclingParser внутри _convert_with_repair: при ошибке Docling он
    один раз пытается починить документ и сконвертировать повторно.
    Поэтому pipeline отдельный repair не делает — он только ловит
    финальную ошибку Docling и переключается на PyMuPDF-парсер.

    Важно:
    Финальные RAG chunks здесь не строим.
    Chunks строятся отдельной стадией после ingestion:
        build_rag_chunks_from_records.py
    """

    def __init__(
        self,
        parser_registry: ParserRegistry,
        storage: LocalJsonlStorage,
        block_cleaner: BlockCleaner | None = None,
        table_extractor: TableExtractor | None = None,
        table_normalizer: TableNormalizer | None = None,
        pdf_renderer: PdfPageRenderer | None = None,
        embedding_exporter: EmbeddingExporter | None = None,
        profiler: DocumentProfiler | None = None,
        router: StrategyRouter | None = None,
        render_visuals: bool = True,
        export_embedding_records: bool = True,
    ) -> None:
        self.parser_registry = parser_registry
        self.storage = storage

        self.block_cleaner = block_cleaner or BlockCleaner()
        self.table_extractor = table_extractor or TableExtractor()
        self.table_normalizer = table_normalizer or TableNormalizer()

        self.pdf_renderer = pdf_renderer or PdfPageRenderer()
        self.embedding_exporter = embedding_exporter or EmbeddingExporter()

        self.profiler = profiler or DocumentProfiler()
        self.router = router or StrategyRouter()

        self.render_visuals = render_visuals
        self.export_embedding_records = export_embedding_records

    def run(
        self,
        path: Path | str,
        project: str | None = None,
        market: str | None = None,
        product: str | None = None,
        language: Language = "unknown",
        source_group: str | None = None,
        manifest_source_type: str | None = None,
        topic: str | None = None,
        strategy_override: str | None = None,
    ) -> ParsedDocument:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        source_type = self._detect_source_type(path)

        metadata = DocumentMetadata(
            source_file=path.name,
            source_path=str(path),
            source_type=source_type,
            project=project,
            market=market,
            product=product,
            language=language,
        )

        # Дополнительные поля из manifest.
        # source_type уже занят техническим типом файла: pdf/docx/txt/xlsx/csv.
        # Поэтому тип из манифеста храним как manifest_source_type.
        metadata.extra["source_group"] = source_group or "general"
        metadata.extra["manifest_source_type"] = manifest_source_type or source_type
        metadata.extra["topic"] = topic

        print("=" * 80)
        print("DOCUMENT PROCESSING PIPELINE")
        print("=" * 80)
        print(f"Input: {path}")
        print(f"Source type: {source_type}")
        print(f"Project: {project}")
        print(f"Market: {market}")
        print(f"Product: {product}")
        print(f"Language: {language}")
        print(f"Source group: {source_group or 'general'}")
        print(f"Manifest source type: {manifest_source_type or source_type}")
        print(f"Topic: {topic}")
        print(f"Render visuals: {self.render_visuals}")
        print(f"Export embedding records: {self.export_embedding_records}")
        print(f"Strategy override: {strategy_override}")
        print("=" * 80)

        selected_strategy: str | None = None

        if source_type == "pdf":
            selected_strategy = self._select_pdf_strategy(
                path=path,
                strategy_override=strategy_override,
                metadata=metadata,
            )
            parser = self._resolve_parser_for_strategy(selected_strategy)
        else:
            parser = self.parser_registry.get_parser(path)
            selected_strategy = getattr(parser, "parser_name", "default")

        metadata.extra["selected_strategy"] = selected_strategy

        document = self._parse_with_fallback(
            parser=parser,
            path=path,
            metadata=metadata,
            source_type=source_type,
            selected_strategy=selected_strategy,
        )

        # На всякий случай обновляем metadata после fallback.
        document.metadata.extra["selected_strategy"] = selected_strategy
        document.metadata.extra["source_group"] = source_group or "general"
        document.metadata.extra["manifest_source_type"] = manifest_source_type or source_type
        document.metadata.extra["topic"] = topic

        # Stage 5 — block cleaning
        document = self.block_cleaner.clean_document(document)

        # Stage 6 — table extraction
        document = self.table_extractor.extract_tables(document)

        # Stage 7 — table normalization
        document = self.table_normalizer.normalize_document(document)

        # Stage 8 — optional visuals
        if self.render_visuals and source_type == "pdf":
            visual_output_dir = Path("data/processed/visual") / path.stem

            document.visuals = self.pdf_renderer.render_to_visuals(
                pdf_path=path,
                output_dir=visual_output_dir,
                metadata=document.metadata,
            )

        # Stage 9 — save parsed document entities
        output_path = self.storage.save(document)

        # Stage 10 — optional embedding records
        embedding_output_path: Path | None = None
        embedding_records_count = 0

        if self.export_embedding_records:
            embedding_records = self.embedding_exporter.export_records(document)
            embedding_records_count = len(embedding_records)

            embedding_output_path = (
                Path("data/processed/embedding_input")
                / f"{path.stem}.embedding_records.jsonl"
            )

            self.embedding_exporter.save_jsonl(
                records=embedding_records,
                output_path=embedding_output_path,
            )

        print("=" * 80)
        print("DOCUMENT PROCESSING DONE")
        print("=" * 80)
        print(f"Parsed document: {path}")
        print(f"Parser: {document.metadata.parser_name}")
        print(f"Strategy: {selected_strategy}")
        print(f"Source group: {document.metadata.extra.get('source_group')}")
        print(f"Manifest source type: {document.metadata.extra.get('manifest_source_type')}")
        print(f"Topic: {document.metadata.extra.get('topic')}")
        print(f"Parser backend: {document.metadata.extra.get('parser_backend')}")
        print(f"Fallback parser: {document.metadata.extra.get('fallback_parser')}")
        print(f"PDF repaired: {document.metadata.extra.get('pdf_repaired')}")
        print(f"Original source path: {document.metadata.extra.get('original_source_path')}")
        print(f"Repaired source path: {document.metadata.extra.get('repaired_source_path')}")
        print(f"Blocks: {len(document.non_empty_blocks())}")
        print(f"Tables: {len(document.non_empty_tables())}")
        print(f"Visuals: {len(document.non_empty_visuals())}")
        print(f"Saved to: {output_path}")

        if self.export_embedding_records:
            print(f"Embedding records: {embedding_records_count}")
            print(f"Embedding input saved to: {embedding_output_path}")

        print("=" * 80)

        return document

    def _select_pdf_strategy(
        self,
        path: Path,
        strategy_override: str | None,
        metadata: DocumentMetadata,
    ) -> DocumentProcessingStrategy:
        """
        Выбирает стратегию обработки PDF.

        Если strategy_override задан, используем его.
        Иначе профилируем документ и выбираем стратегию через StrategyRouter.
        """

        if strategy_override:
            normalized = self._normalize_strategy_override(strategy_override)
            metadata.extra["strategy_override"] = strategy_override
            return normalized

        profile = self.profiler.profile(path)
        strategy = self.router.select(profile)

        print(self.router.explain(profile))

        metadata.extra["document_profile"] = {
            "file_name": profile.file_name,
            "suffix": profile.suffix,
            "file_size_mb": profile.file_size_mb,
            "page_count": profile.page_count,
            "can_open_with_pymupdf": profile.can_open_with_pymupdf,
            "pymupdf_error": profile.pymupdf_error,
            "has_text_layer": profile.has_text_layer,
            "total_text_chars": profile.total_text_chars,
            "avg_chars_per_page": profile.avg_chars_per_page,
            "text_pages": profile.text_pages,
            "text_page_ratio": profile.text_page_ratio,
            "table_likeness": profile.table_likeness,
            "numeric_line_ratio": profile.numeric_line_ratio,
            "table_keyword_hits": profile.table_keyword_hits,
            "image_count": profile.image_count,
            "avg_images_per_page": profile.avg_images_per_page,
            "is_large_document": profile.is_large_document,
            "is_mostly_scanned": profile.is_mostly_scanned,
            "is_table_heavy": profile.is_table_heavy,
            "recommended_strategy": profile.recommended_strategy,
            "reasons": profile.reasons,
        }

        return strategy

    @staticmethod
    def _normalize_strategy_override(strategy: str) -> DocumentProcessingStrategy:
        """
        Нормализует ручной parser/strategy override.

        Поддерживаем:
        - docling
        - pymupdf
        - docling_full
        - pymupdf_text
        - pymupdf_table_like
        - hybrid_docling_then_pymupdf
        - ocr_required
        """

        mapping: dict[str, DocumentProcessingStrategy] = {
            "docling": "docling_full",
            "pymupdf": "pymupdf_text",
            "docling_full": "docling_full",
            "pymupdf_text": "pymupdf_text",
            "pymupdf_table_like": "pymupdf_table_like",
            "hybrid_docling_then_pymupdf": "hybrid_docling_then_pymupdf",
            "ocr_required": "ocr_required",
        }

        if strategy not in mapping:
            raise ValueError(f"Unknown strategy override: {strategy}")

        return mapping[strategy]

    def _resolve_parser_for_strategy(
        self,
        strategy: str,
    ) -> BaseParser:
        """
        Маппинг strategy → parser.

        Важно:
        Даже если выбрали docling_full, при ошибке Docling сработает
        fallback на PyMuPDF (см. _parse_with_fallback). Сам repair PDF
        делает PdfDoclingParser внутри _convert_with_repair.
        """

        if strategy == "docling_full":
            return self.parser_registry.get_by_name("docling")

        if strategy in {"pymupdf_text", "pymupdf_table_like"}:
            return self.parser_registry.get_by_name("pymupdf")

        if strategy == "hybrid_docling_then_pymupdf":
            try:
                return self.parser_registry.get_by_name("hybrid_docling_pymupdf")
            except Exception:
                return self.parser_registry.get_by_name("docling")

        if strategy == "ocr_required":
            # Слот "ocr_required" теперь обслуживает VLM-парсер (Qwen3-VL
            # через Ollama). Если он по какой-то причине не зарегистрирован
            # (нет Ollama, нет модели), мягко откатываемся на PyMuPDF.
            try:
                return self.parser_registry.get_by_name("vlm")
            except Exception:
                print("=" * 80)
                print("OCR/VLM STRATEGY SELECTED, BUT VLM PARSER NOT FOUND")
                print("FALLBACK TO PYMUPDF")
                print("=" * 80)
                return self.parser_registry.get_by_name("pymupdf")

        raise ValueError(f"Cannot resolve parser for strategy: {strategy}")

    def _parse_with_fallback(
        self,
        parser: BaseParser,
        path: Path,
        metadata: DocumentMetadata,
        source_type: SourceType,
        selected_strategy: str | None,
    ) -> ParsedDocument:
        """
        Запускает выбранный parser.

        Для PDF + Docling:
        1. Пробуем Docling. Сам PdfDoclingParser при ошибке один раз
           чинит PDF (repair через PyMuPDF) и пробует сконвертировать
           повторно — это происходит внутри parser.parse().
        2. Если Docling всё равно упал — fallback на PyMuPDF-парсер.

        Для остальных parser-ов:
        - пробуем parser;
        - если упал, исключение поднимается выше.

        Отдельный repair на уровне pipeline убран намеренно: он повторял
        ту же самую операцию (пересохранение через PyMuPDF), что уже
        делает PdfDoclingParser, и при этом затирал metadata.source_path.
        """

        parser_name = getattr(parser, "parser_name", "unknown")
        parser_version = getattr(parser, "parser_version", "unknown")

        print("=" * 80)
        print(f"Using parser: {parser_name} (v{parser_version})")
        print("=" * 80)

        try:
            document = parser.parse(path=path, metadata=metadata)
            document.metadata.extra["selected_strategy"] = selected_strategy
            document.metadata.extra["parser_backend"] = parser_name
            # Docling-парсер мог сам починить PDF внутри _convert_with_repair.
            document.metadata.extra.setdefault(
                "pdf_repaired",
                bool(document.metadata.extra.get("docling_used_repaired_pdf", False)),
            )
            return document

        except Exception as exc:
            if source_type == "pdf" and parser_name == "docling":
                first_error = f"{type(exc).__name__}: {exc}"

                print("=" * 80)
                print("DOCLING FAILED — FALLBACK TO PYMUPDF")
                print("=" * 80)
                print(first_error)

                metadata.extra["docling_failure"] = first_error

                fallback_parser = self.parser_registry.get_by_name("pymupdf")

                fallback_name = getattr(fallback_parser, "parser_name", "pymupdf")
                fallback_version = getattr(fallback_parser, "parser_version", "unknown")

                print("=" * 80)
                print(f"Using fallback parser: {fallback_name} (v{fallback_version})")
                print("=" * 80)

                document = fallback_parser.parse(path=path, metadata=metadata)

                document.metadata.extra["docling_failure"] = first_error
                document.metadata.extra["fallback_parser"] = fallback_name
                document.metadata.extra["selected_strategy"] = selected_strategy
                document.metadata.extra["parser_backend"] = fallback_name

                return document

            raise

    @staticmethod
    def _detect_source_type(path: Path) -> SourceType:
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return "pdf"
        if suffix == ".docx":
            return "docx"
        if suffix == ".txt":
            return "txt"
        if suffix in {".html", ".htm"}:
            return "html"
        if suffix == ".xlsx":
            return "xlsx"
        if suffix == ".csv":
            return "csv"

        return "unknown"