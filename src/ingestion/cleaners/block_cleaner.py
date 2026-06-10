from __future__ import annotations

import re

from ingestion.models import ParsedBlock, ParsedDocument


class BlockCleaner:
    """
    Базовая очистка и разметка ParsedBlock после parser backend.

    На этом этапе мы НЕ удаляем агрессивно данные.
    Мы:
    - нормализуем пробелы;
    - убираем пустые блоки;
    - помечаем image_placeholder как skip_embedding;
    - помечаем таблицы;
    - помечаем слабые/короткие блоки.
    """

    min_text_length: int = 20

    def clean_document(self, document: ParsedDocument) -> ParsedDocument:
        cleaned_blocks: list[ParsedBlock] = []

        for block in document.blocks:
            cleaned = self.clean_block(block)

            if cleaned is None:
                continue

            cleaned_blocks.append(cleaned)

        document.blocks = cleaned_blocks
        return document

    def clean_block(self, block: ParsedBlock) -> ParsedBlock | None:
        text = self._normalize_text(block.block_text)

        if not text:
            return None

        block.block_text = text

        # Технические блоки картинок не должны идти в embeddings.
        if block.block_type == "image_placeholder":
            block.metadata["skip_embedding"] = True
            block.metadata["reason"] = "image placeholder from document parser"

        # Таблицы отдельно помечаем для будущей обработки.
        if block.block_type == "table":
            block.metadata["is_table"] = True
            block.metadata["needs_table_processing"] = True

        # Очень короткие блоки лучше не эмбеддить без контекста.
        # Но не перезаписываем reason для image_placeholder.
        if (
                len(text) < self.min_text_length
                and block.block_type not in {"table", "image_placeholder"}
        ):
            block.metadata["low_value"] = True
            block.metadata["skip_embedding"] = True
            block.metadata["reason"] = "too short text block"

        return block

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\x00", " ")
        text = text.replace("\u00a0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()