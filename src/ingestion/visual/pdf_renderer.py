from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from ingestion.models import DocumentMetadata, ParsedVisual


class PdfPageRenderer:
    """
    Рендерит страницы PDF в PNG и создает ParsedVisual.

    На первом этапе сохраняем всю страницу как изображение.
    Позже можно будет добавить crop отдельных графиков/таблиц.
    """

    renderer_name: str = "pymupdf_page_renderer"
    renderer_version: str = "0.1.0"

    def __init__(self, dpi: int = 200) -> None:
        self.dpi = dpi

    def render_to_visuals(
        self,
        pdf_path: Path | str,
        output_dir: Path | str,
        metadata: DocumentMetadata,
        pages: list[int] | None = None,
    ) -> list[ParsedVisual]:
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        visuals: list[ParsedVisual] = []

        zoom = self.dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        visual_order = 0

        for page_index in range(len(doc)):
            page_number = page_index + 1

            if pages is not None and page_number not in pages:
                continue

            page = doc[page_index]
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            image_name = f"{pdf_path.stem}_page_{page_number:03d}.png"
            image_path = output_dir / image_name

            pix.save(image_path)

            visual = ParsedVisual(
                image_path=str(image_path),
                visual_type="page_image",
                visual_order=visual_order,
                source_file=metadata.source_file,
                source_type=metadata.source_type,
                page_number=page_number,
                caption=None,
                ocr_text=None,
                description=None,
                language=metadata.language,
                parser_name=self.renderer_name,
                parser_version=self.renderer_version,
                metadata={
                    "project": metadata.project,
                    "market": metadata.market,
                    "product": metadata.product,
                    "needs_visual_processing": True,
                    "visual_source": "rendered_pdf_page",
                    "dpi": self.dpi,
                },
            )

            visuals.append(visual)
            visual_order += 1

        doc.close()

        return visuals