from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

from ingestion.models import ParsedDocument


@dataclass
class EmbeddingRecord:
    """
    Единица текста, которую можно отправлять в embedding-модель.

    Это уже не сырая сущность документа, а подготовленная запись для RAG.
    """

    record_id: str
    source_file: str
    source_type: str
    entity_type: str

    text: str

    page_number: int | None = None
    entity_order: int | None = None

    project: str | None = None
    market: str | None = None
    product: str | None = None
    language: str = "unknown"

    metadata: dict | None = None


class EmbeddingExporter:
    """
    Готовит данные для embeddings.

    Правило:
    - текстовые blocks берем, если skip_embedding != True;
    - таблицы берем через table.metadata["table_text"];
    - visuals пока не берем, если у них нет description / ocr_text.
    """

    def export_records(self, document: ParsedDocument) -> list[EmbeddingRecord]:
        records: list[EmbeddingRecord] = []

        records.extend(self._export_blocks(document))
        records.extend(self._export_tables(document))
        records.extend(self._export_visuals(document))

        records.sort(
            key=lambda record: (
                record.page_number if record.page_number is not None else 10 ** 9,
                record.entity_order if record.entity_order is not None else 10 ** 9,
                record.entity_type,
            )
        )

        return records

    def save_jsonl(
        self,
        records: list[EmbeddingRecord],
        output_path: Path | str,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        return output_path

    def _base_document_metadata(self, document: ParsedDocument) -> dict:
        """
        Metadata документа, которую надо протащить в каждый embedding record.

        Важно:
        - source_type в корне EmbeddingRecord уже есть.
        - manifest_source_type храним отдельно, чтобы не путать с техническим типом файла.
        - document.metadata.extra может содержать document_profile; он бывает большим,
          поэтому его не тащим в каждый record.
        """

        extra = document.metadata.extra or {}

        return {
            "source_path": document.metadata.source_path,
            "source_group": extra.get("source_group", "general"),
            "manifest_source_type": extra.get("manifest_source_type", document.metadata.source_type),
            "topic": extra.get("topic"),
            "selected_strategy": extra.get("selected_strategy"),
            "parser_backend": extra.get("parser_backend"),
            "fallback_parser": extra.get("fallback_parser"),
            "docling_failure": extra.get("docling_failure"),
        }

    def _export_blocks(self, document: ParsedDocument) -> list[EmbeddingRecord]:
        records: list[EmbeddingRecord] = []

        doc_meta = self._base_document_metadata(document)

        for block in document.non_empty_blocks():
            if block.metadata.get("skip_embedding") is True:
                continue

            text = block.block_text.strip()
            if not text:
                continue

            record = EmbeddingRecord(
                record_id=f"{document.metadata.source_file}:block:{block.block_order}",
                source_file=document.metadata.source_file,
                source_type=document.metadata.source_type,
                entity_type=f"block:{block.block_type}",
                text=text,
                page_number=block.page_number,
                entity_order=block.block_order,
                project=document.metadata.project,
                market=document.metadata.market,
                product=document.metadata.product,
                language=block.language,
                metadata={
                    **doc_meta,
                    **block.metadata,
                    "parser_name": block.parser_name,
                    "parser_version": block.parser_version,
                },
            )

            records.append(record)

        return records

    def _export_tables(self, document: ParsedDocument) -> list[EmbeddingRecord]:
        records: list[EmbeddingRecord] = []

        doc_meta = self._base_document_metadata(document)

        for table in document.non_empty_tables():
            quality = table.metadata.get("table_quality")

            if quality == "bad":
                continue

            table_text = table.metadata.get("table_text")
            if not table_text:
                table_text = table.markdown_table

            table_text = table_text.strip()
            if not table_text:
                continue

            record = EmbeddingRecord(
                record_id=f"{document.metadata.source_file}:table:{table.table_order}",
                source_file=document.metadata.source_file,
                source_type=document.metadata.source_type,
                entity_type="table",
                text=table_text,
                page_number=table.page_number,
                entity_order=table.metadata.get("extracted_from_block_order", table.table_order),
                project=document.metadata.project,
                market=document.metadata.market,
                product=document.metadata.product,
                language=table.language,
                metadata={
                    **doc_meta,
                    # section_path для таблиц раньше терялся — теперь
                    # пробрасываем явно (Docling/VLM кладут его в
                    # table.metadata, у XlsxParser он совпадает с
                    # sheet_name).
                    "section_path": table.metadata.get("section_path"),
                    "table_quality": quality,
                    "columns": table.metadata.get("columns"),
                    "normalized_rows": table.metadata.get("normalized_rows"),
                    "parser_name": table.parser_name,
                    "parser_version": table.parser_version,
                },
            )

            records.append(record)

        return records

    def _export_visuals(self, document: ParsedDocument) -> list[EmbeddingRecord]:
        """
        Пока visuals почти всегда не идут в embeddings.

        В будущем сюда попадут:
        - visual.description
        - visual.ocr_text
        - VLM summary графика
        """

        records: list[EmbeddingRecord] = []

        doc_meta = self._base_document_metadata(document)

        for visual in document.non_empty_visuals():
            visual_text = None

            if visual.description and visual.description.strip():
                visual_text = visual.description.strip()
            elif visual.ocr_text and visual.ocr_text.strip():
                visual_text = visual.ocr_text.strip()

            if not visual_text:
                continue

            record = EmbeddingRecord(
                record_id=f"{document.metadata.source_file}:visual:{visual.visual_order}",
                source_file=document.metadata.source_file,
                source_type=document.metadata.source_type,
                entity_type=f"visual:{visual.visual_type}",
                text=visual_text,
                page_number=visual.page_number,
                entity_order=visual.visual_order,
                project=document.metadata.project,
                market=document.metadata.market,
                product=document.metadata.product,
                language=visual.language,
                metadata={
                    **doc_meta,
                    **visual.metadata,
                    "image_path": visual.image_path,
                    "parser_name": visual.parser_name,
                    "parser_version": visual.parser_version,
                },
            )

            records.append(record)

        return records