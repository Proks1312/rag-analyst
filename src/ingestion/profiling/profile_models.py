from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


DocumentProcessingStrategy = Literal[
    "docling_full",
    "pymupdf_text",
    "pymupdf_table_like",
    "hybrid_docling_then_pymupdf",
    "ocr_required",
]


@dataclass
class DocumentProfile:
    path: Path
    file_name: str
    suffix: str
    file_size_mb: float

    page_count: int | None = None

    can_open_with_pymupdf: bool = False
    pymupdf_error: str | None = None

    has_text_layer: bool = False
    total_text_chars: int = 0
    avg_chars_per_page: float = 0.0
    text_pages: int = 0
    text_page_ratio: float = 0.0

    table_likeness: float = 0.0
    numeric_line_ratio: float = 0.0
    table_keyword_hits: int = 0

    image_count: int = 0
    avg_images_per_page: float = 0.0

    is_large_document: bool = False
    is_mostly_scanned: bool = False
    is_table_heavy: bool = False

    recommended_strategy: DocumentProcessingStrategy = "hybrid_docling_then_pymupdf"
    reasons: list[str] = field(default_factory=list)