from __future__ import annotations

from ingestion.profiling.profile_models import DocumentProfile, DocumentProcessingStrategy


class StrategyRouter:
    """
    Выбор стратегии обработки на основе DocumentProfile.
    """

    def select(self, profile: DocumentProfile) -> DocumentProcessingStrategy:
        return profile.recommended_strategy

    def explain(self, profile: DocumentProfile) -> str:
        lines = [
            "DOCUMENT PROFILE",
            "=" * 80,
            f"file_name: {profile.file_name}",
            f"suffix: {profile.suffix}",
            f"file_size_mb: {profile.file_size_mb:.2f}",
            f"page_count: {profile.page_count}",
            f"can_open_with_pymupdf: {profile.can_open_with_pymupdf}",
            f"has_text_layer: {profile.has_text_layer}",
            f"total_text_chars_sample: {profile.total_text_chars}",
            f"avg_chars_per_page_sample: {profile.avg_chars_per_page:.1f}",
            f"text_page_ratio_sample: {profile.text_page_ratio:.2f}",
            f"numeric_line_ratio_sample: {profile.numeric_line_ratio:.2f}",
            f"table_keyword_hits_sample: {profile.table_keyword_hits}",
            f"table_likeness: {profile.table_likeness:.2f}",
            f"image_count_sample: {profile.image_count}",
            f"avg_images_per_page_sample: {profile.avg_images_per_page:.2f}",
            f"is_large_document: {profile.is_large_document}",
            f"is_mostly_scanned: {profile.is_mostly_scanned}",
            f"is_table_heavy: {profile.is_table_heavy}",
            f"recommended_strategy: {profile.recommended_strategy}",
            "reasons:",
        ]

        for reason in profile.reasons:
            lines.append(f"  - {reason}")

        lines.append("=" * 80)

        return "\n".join(lines)