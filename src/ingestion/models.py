from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


SourceType = Literal[
    "pdf",
    "docx",
    "txt",
    "html",
    "xlsx",
    "csv",
    "unknown",
]

BlockType = Literal[
    "title",
    "heading",
    "paragraph",
    "table",
    "list",
    "image_placeholder",
    "chart_text",
    "note",
    "raw_text",
    "unknown",
]

VisualType = Literal[
    "page_image",
    "chart",
    "figure",
    "scanned_table",
    "image",
    "unknown",
]

Language = Literal[
    "ru",
    "en",
    "zh",
    "mixed",
    "unknown",
]


@dataclass
class DocumentMetadata:
    """
    Метаданные документа.

    Это информация уровня всего файла:
    откуда файл, к какому проекту/рынку/продукту относится,
    какой язык, какой парсер использовался.
    """

    source_file: str
    source_path: str
    source_type: SourceType

    project: str | None = None
    market: str | None = None
    product: str | None = None

    language: Language = "unknown"
    title: str | None = None
    source_url: str | None = None
    document_date: str | None = None

    content_hash: str | None = None
    parser_name: str | None = None
    parser_version: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedBlock:
    """
    Один обработанный блок документа после prechunking.

    Это еще НЕ финальный chunk для embeddings.
    Это промежуточный смысловой блок:
    заголовок, абзац, таблица, список, placeholder изображения и т.д.
    """

    block_text: str
    block_type: BlockType

    block_order: int
    source_file: str
    source_type: SourceType

    page_number: int | None = None
    section_title: str | None = None
    language: Language = "unknown"

    parser_name: str | None = None
    parser_version: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.block_text or not self.block_text.strip()


@dataclass
class ParsedTable:
    """
    Отдельная сущность для таблиц.

    Таблицы лучше не смешивать с обычным текстом:
    их надо хранить отдельно в raw/json и markdown-представлении.
    """

    markdown_table: str
    raw_data: list[list[Any]]

    table_order: int
    source_file: str
    source_type: SourceType

    page_number: int | None = None
    table_title: str | None = None
    language: Language = "unknown"

    parser_name: str | None = None
    parser_version: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.markdown_table or not self.markdown_table.strip()


@dataclass
class ParsedVisual:
    """
    Отдельная сущность для изображений и визуальных элементов.

    На первом этапе будем сохранять изображения страниц PDF.
    Позже сюда можно добавлять OCR-текст, описание графика,
    тип визуального объекта и результат VLM/OCR-обработки.
    """

    image_path: str
    visual_type: VisualType

    visual_order: int
    source_file: str
    source_type: SourceType

    page_number: int | None = None
    caption: str | None = None
    ocr_text: str | None = None
    description: str | None = None
    language: Language = "unknown"

    parser_name: str | None = None
    parser_version: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def has_text(self) -> bool:
        return bool(
            (self.caption and self.caption.strip())
            or (self.ocr_text and self.ocr_text.strip())
            or (self.description and self.description.strip())
        )


@dataclass
class ParsedDocument:
    """
    Результат работы любого parser backend.

    Любой парсер должен возвращать ParsedDocument:
    Docling, PyMuPDF, DOCX parser, HTML parser, Excel parser и т.д.
    """

    metadata: DocumentMetadata
    blocks: list[ParsedBlock] = field(default_factory=list)
    tables: list[ParsedTable] = field(default_factory=list)
    visuals: list[ParsedVisual] = field(default_factory=list)

    def non_empty_blocks(self) -> list[ParsedBlock]:
        return [block for block in self.blocks if not block.is_empty()]

    def non_empty_tables(self) -> list[ParsedTable]:
        return [table for table in self.tables if not table.is_empty()]

    def non_empty_visuals(self) -> list[ParsedVisual]:
        return [visual for visual in self.visuals if visual.image_path]

    @property
    def source_path(self) -> Path:
        return Path(self.metadata.source_path)