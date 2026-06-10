from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from .profile_models import DocumentProfile


class DocumentProfiler:
    """
    Быстрый профилировщик документа.

    Задача:
    - не делать тяжелый OCR;
    - не гонять Docling;
    - быстро понять, есть ли текстовый слой, сколько страниц,
      похож ли документ на табличный, большой ли он.
    """

    def __init__(
        self,
        sample_pages: int = 10,
        large_document_pages: int = 80,
    ) -> None:
        self.sample_pages = sample_pages
        self.large_document_pages = large_document_pages

    def profile(self, path: Path | str) -> DocumentProfile:
        path = Path(path)

        profile = DocumentProfile(
            path=path,
            file_name=path.name,
            suffix=path.suffix.lower(),
            file_size_mb=self._file_size_mb(path),
        )

        if profile.suffix != ".pdf":
            profile.reasons.append(f"Unsupported or non-PDF suffix: {profile.suffix}")
            profile.recommended_strategy = "hybrid_docling_then_pymupdf"
            return profile

        try:
            self._profile_pdf_with_pymupdf(path, profile)
        except Exception as exc:
            profile.can_open_with_pymupdf = False
            profile.pymupdf_error = f"{type(exc).__name__}: {exc}"
            profile.reasons.append(f"PyMuPDF failed: {profile.pymupdf_error}")
            profile.recommended_strategy = "ocr_required"
            return profile

        self._derive_flags(profile)
        self._recommend_strategy(profile)

        return profile

    def _profile_pdf_with_pymupdf(self, path: Path, profile: DocumentProfile) -> None:
        doc = fitz.open(path)

        try:
            profile.can_open_with_pymupdf = True
            profile.page_count = doc.page_count

            sample_page_indexes = self._sample_page_indexes(doc.page_count)

            total_chars = 0
            text_pages = 0
            numeric_lines = 0
            total_lines = 0
            table_keyword_hits = 0
            image_count = 0

            for page_index in sample_page_indexes:
                page = doc.load_page(page_index)

                text = page.get_text("text") or ""
                clean_text = self._clean_text(text)

                char_count = len(clean_text)
                total_chars += char_count

                if char_count >= 50:
                    text_pages += 1

                lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
                total_lines += len(lines)

                for line in lines:
                    if self._looks_numeric_line(line):
                        numeric_lines += 1

                table_keyword_hits += self._count_table_keywords(clean_text)

                try:
                    image_count += len(page.get_images(full=True))
                except Exception:
                    pass

            sampled_pages = max(1, len(sample_page_indexes))

            profile.total_text_chars = total_chars
            profile.text_pages = text_pages
            profile.avg_chars_per_page = total_chars / sampled_pages
            profile.text_page_ratio = text_pages / sampled_pages
            profile.has_text_layer = profile.text_page_ratio >= 0.3 and profile.avg_chars_per_page >= 100

            profile.numeric_line_ratio = numeric_lines / max(1, total_lines)
            profile.table_keyword_hits = table_keyword_hits

            # Простая универсальная оценка "табличности".
            table_score = 0.0

            if profile.numeric_line_ratio >= 0.25:
                table_score += 0.45
            elif profile.numeric_line_ratio >= 0.15:
                table_score += 0.25

            if table_keyword_hits >= 5:
                table_score += 0.35
            elif table_keyword_hits >= 2:
                table_score += 0.20

            if profile.avg_chars_per_page >= 1000:
                table_score += 0.10

            profile.table_likeness = min(1.0, table_score)

            profile.image_count = image_count
            profile.avg_images_per_page = image_count / sampled_pages

        finally:
            doc.close()

    def _derive_flags(self, profile: DocumentProfile) -> None:
        page_count = profile.page_count or 0

        profile.is_large_document = page_count >= self.large_document_pages

        profile.is_mostly_scanned = (
            not profile.has_text_layer
            and profile.avg_images_per_page >= 0.5
        )

        profile.is_table_heavy = (
            profile.table_likeness >= 0.45
            or profile.numeric_line_ratio >= 0.25
            or profile.table_keyword_hits >= 5
        )

    def _recommend_strategy(self, profile: DocumentProfile) -> None:
        if not profile.can_open_with_pymupdf:
            profile.recommended_strategy = "ocr_required"
            profile.reasons.append("Cannot open with PyMuPDF; OCR or repair may be required.")
            return

        if profile.is_mostly_scanned:
            profile.recommended_strategy = "ocr_required"
            profile.reasons.append("Document looks scanned: weak text layer and many images.")
            return

        if profile.is_large_document and profile.has_text_layer and profile.is_table_heavy:
            profile.recommended_strategy = "pymupdf_table_like"
            profile.reasons.append("Large table-heavy PDF with text layer; avoid heavy Docling preprocessing.")
            return

        if profile.has_text_layer and profile.is_table_heavy and (profile.page_count or 0) > 50:
            profile.recommended_strategy = "pymupdf_table_like"
            profile.reasons.append("Table-heavy PDF with many pages; PyMuPDF table-like strategy is safer.")
            return

        if profile.has_text_layer and not profile.is_table_heavy and (profile.page_count or 0) > 80:
            profile.recommended_strategy = "pymupdf_text"
            profile.reasons.append("Large text-layer PDF; PyMuPDF text strategy is safer.")
            return

        if (profile.page_count or 0) <= 60:
            profile.recommended_strategy = "docling_full"
            profile.reasons.append("Small/medium PDF; Docling full strategy is acceptable.")
            return

        profile.recommended_strategy = "hybrid_docling_then_pymupdf"
        profile.reasons.append("Default strategy: try Docling, fallback to PyMuPDF.")

    def _sample_page_indexes(self, page_count: int) -> list[int]:
        if page_count <= self.sample_pages:
            return list(range(page_count))

        indexes = {0, page_count - 1}

        # равномерная выборка
        for i in range(1, self.sample_pages - 1):
            idx = int(i * (page_count - 1) / (self.sample_pages - 1))
            indexes.add(idx)

        return sorted(indexes)

    @staticmethod
    def _file_size_mb(path: Path) -> float:
        if not path.exists():
            return 0.0
        return path.stat().st_size / 1024 / 1024

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\x00", " ")
        text = text.replace("\u00a0", " ")
        text = text.replace("\ufeff", " ")
        text = text.replace("\xad", "")
        return text.strip()

    @staticmethod
    def _looks_numeric_line(line: str) -> bool:
        numbers = re.findall(r"-?\d+(?:[.,]\d+)?", line)

        if len(numbers) >= 4:
            return True

        years = re.findall(r"\b20\d{2}\b", line)

        if len(years) >= 3:
            return True

        return False

    @staticmethod
    def _count_table_keywords(text: str) -> int:
        norm = text.lower().replace("ё", "е")

        keywords = [
            "таблица",
            "наименование показателя",
            "количество пожаров",
            "прямой ущерб",
            "погибло",
            "травмировано",
            "2020",
            "2021",
            "2022",
            "2023",
            "2024",
        ]

        return sum(1 for keyword in keywords if keyword in norm)