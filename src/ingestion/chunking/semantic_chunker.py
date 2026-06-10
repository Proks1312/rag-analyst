from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from ingestion.exporters.embedding_exporter import EmbeddingRecord


# ============================================================
# DATA MODEL
# ============================================================


@dataclass
class RagChunk:
    """
    Финальный текстовый фрагмент для embeddings / RAG.

    ВАЖНО про имена полей:
    Поля section_title и page_number названы так намеренно — именно их
    ожидают downstream-скрипты load_rag_chunks_to_db.py и answer_question.py.
    section_title здесь содержит полный путь "Раздел / Подраздел / Подпункт"
    (это нормально — путь и есть заголовок секции).

    Поля:
    - chunk_id          стабильный id вида "<source_file>:rag_chunk:<order>".
    - chunk_type        text | list | table | visual.
    - text              текст для эмбеддинга (с приклеенной section-шапкой).
    - section_title     полный путь секции (для loader / rerank).
    - page_number       первая (минимальная) страница чанка — для loader.
    - page_numbers      полный список страниц, которые покрывает чанк.
    - source_record_ids id исходных EmbeddingRecord, из которых собран чанк.
    """

    chunk_id: str
    source_file: str
    source_type: str
    chunk_type: str

    text: str

    section_title: Optional[str] = None
    page_number: Optional[int] = None
    page_numbers: list[int] = field(default_factory=list)
    chunk_order: int = 0

    project: Optional[str] = None
    market: Optional[str] = None
    product: Optional[str] = None
    language: str = "unknown"

    source_record_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ============================================================
# CHUNKER
# ============================================================


class SemanticChunker:
    """
    Структурный чанкер для RAG.

    Принципы:
    1. Чанк = последовательность блоков с одинаковым section_path,
       которая укладывается в target_chunk_chars и не превышает max_chunk_chars.
    2. Таблицы — каждая отдельным чанком, не объединяем с текстом и не дробим.
    3. Visual-записи (с описанием/OCR) — отдельным чанком.
    4. При переходе section_path → emit текущий буфер.
    5. На границе чанков — overlap по последним overlap_sentences предложениям.
    6. Очень короткие изолированные блоки (< min_chunk_chars) приклеиваем
       к следующему блоку в том же section_path, если возможно.
    7. Каждый чанк получает в начале текстовую "шапку" с section_path
       и source_file — это и помогает поиску, и даёт LLM контекст.
    """

    def __init__(
        self,
        target_chunk_chars: int = 1500,
        max_chunk_chars: int = 3000,
        min_chunk_chars: int = 200,
        overlap_sentences: int = 1,
        prepend_section_header: bool = True,
    ) -> None:
        if max_chunk_chars < target_chunk_chars:
            raise ValueError("max_chunk_chars must be >= target_chunk_chars")
        if min_chunk_chars >= target_chunk_chars:
            raise ValueError("min_chunk_chars must be < target_chunk_chars")

        self.target_chunk_chars = target_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.overlap_sentences = max(0, overlap_sentences)
        self.prepend_section_header = prepend_section_header

    # ============================================================
    # PUBLIC
    # ============================================================

    def chunk_records(self, records: list[EmbeddingRecord]) -> list[RagChunk]:
        if not records:
            return []

        # Нормализуем текст один раз на входе. Делаем перед чанкингом,
        # чтобы и сплиттер таблиц, и текстовые чанки работали уже на
        # чистом тексте без <br>, разорванных переносами слов и лишних
        # пробелов. Эмбеддинг получает чистые токены.
        for record in records:
            if record.text:
                record.text = self._normalize_text(record.text)

        chunks: list[RagChunk] = []
        chunk_order = 0

        # records уже отсортированы exporter'ом по page_number / entity_order.
        # Но всё равно сделаем стабильную группировку по source_file.
        for source_file, group in self._group_by_source(records):
            file_chunks, chunk_order = self._chunk_one_document(
                source_file=source_file,
                records=group,
                start_order=chunk_order,
            )
            chunks.extend(file_chunks)

        return chunks

    def save_jsonl(
        self,
        chunks: list[RagChunk],
        output_path: Path | str,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

        return output_path

    @staticmethod
    def load_records_jsonl(path: Path | str) -> list[EmbeddingRecord]:
        path = Path(path)
        records: list[EmbeddingRecord] = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append(EmbeddingRecord(**obj))

        return records

    # ============================================================
    # DOCUMENT-LEVEL CHUNKING
    # ============================================================

    def _chunk_one_document(
        self,
        source_file: str,
        records: list[EmbeddingRecord],
        start_order: int,
    ) -> tuple[list[RagChunk], int]:
        chunks: list[RagChunk] = []
        chunk_order = start_order

        # Буфер для накопления текстовых блоков одной секции.
        buffer_text_parts: list[str] = []
        buffer_records: list[EmbeddingRecord] = []
        buffer_pages: set[int] = set()
        buffer_section: Optional[str] = None

        def emit_buffer(chunk_type: str = "text") -> None:
            nonlocal buffer_text_parts, buffer_records, buffer_pages
            nonlocal buffer_section, chunk_order

            if not buffer_records:
                return

            joined = self._join_parts(buffer_text_parts)
            if not joined.strip():
                buffer_text_parts = []
                buffer_records = []
                buffer_pages = set()
                buffer_section = None
                return

            # Отбрасываем крошечные текстовые чанки (короче min_chunk_chars):
            # это обычно обрывки предложений или одиночные ярлыки, в
            # retrieval только зашумляют. Таблицы/visuals идут отдельным
            # путём и сюда не попадают.
            if len(joined.strip()) < self.min_chunk_chars:
                buffer_text_parts = []
                buffer_records = []
                buffer_pages = set()
                return

            chunk = self._make_chunk(
                chunk_order=chunk_order,
                chunk_type=chunk_type,
                text=joined,
                section_path=buffer_section,
                source_records=buffer_records,
                pages=buffer_pages,
            )
            chunks.append(chunk)
            chunk_order += 1

            # Подготовка overlap для следующего чанка.
            overlap_text = self._overlap_text(joined)
            buffer_text_parts = [overlap_text] if overlap_text else []
            buffer_records = []
            buffer_pages = set()
            # section не сбрасываем — overlap считается продолжением той же секции.

        for record in records:
            entity_type = record.entity_type or ""

            # 1) Таблицы — отдельный чанк (или несколько, если таблица
            # большая и не лезет в max_chunk_chars). Передаём текущий
            # buffer_section как fallback — если у самой табличной записи
            # нет section_path в metadata, возьмём раздел из окружающего
            # текста; так таблицы перестают вылетать без контекста.
            if entity_type == "table":
                emit_buffer()
                table_chunks_list = self._make_table_chunks(
                    start_order=chunk_order,
                    record=record,
                    fallback_section=buffer_section,
                )
                chunks.extend(table_chunks_list)
                chunk_order += len(table_chunks_list)
                buffer_text_parts = []
                buffer_records = []
                buffer_pages = set()
                buffer_section = None
                continue

            # 2) Visual с текстовым описанием — отдельный чанк.
            if entity_type.startswith("visual:"):
                emit_buffer()
                visual_chunk = self._make_visual_chunk(
                    chunk_order=chunk_order,
                    record=record,
                )
                chunks.append(visual_chunk)
                chunk_order += 1
                buffer_text_parts = []
                buffer_records = []
                buffer_pages = set()
                buffer_section = None
                continue

            # 3) Текстовые блоки (paragraph / list / title / прочее).
            text = (record.text or "").strip()
            if not text:
                continue

            record_section = self._record_section(record)

            # Если у нас была накоплена другая секция — flush.
            if buffer_section is not None and record_section != buffer_section:
                emit_buffer()
                buffer_section = record_section

            if buffer_section is None:
                buffer_section = record_section

            current_size = sum(len(p) for p in buffer_text_parts)
            added_size = current_size + len(text) + 2

            # Если уже не помещается — flush, потом добавим текущий блок к новому.
            if buffer_text_parts and added_size > self.max_chunk_chars:
                emit_buffer()

            buffer_text_parts.append(text)
            buffer_records.append(record)
            if record.page_number is not None:
                buffer_pages.add(int(record.page_number))

            current_size = sum(len(p) for p in buffer_text_parts)
            if current_size >= self.target_chunk_chars and len(buffer_records) > 0:
                emit_buffer()

        emit_buffer()

        return chunks, chunk_order

    # ============================================================
    # CHUNK FACTORIES
    # ============================================================

    def _make_chunk(
        self,
        chunk_order: int,
        chunk_type: str,
        text: str,
        section_path: Optional[str],
        source_records: list[EmbeddingRecord],
        pages: set[int],
    ) -> RagChunk:
        ref_record = source_records[0]
        header_text = self._with_context_header(
            text=text,
            source_file=ref_record.source_file,
            source_group=(ref_record.metadata or {}).get("source_group"),
            section_path=section_path,
        )

        merged_record_meta = self._merge_record_metadata(source_records)

        return RagChunk(
            chunk_id=f"{ref_record.source_file}:rag_chunk:{chunk_order:05d}",
            source_file=ref_record.source_file,
            source_type=ref_record.source_type,
            chunk_type=chunk_type,
            text=header_text,
            section_title=section_path,
            page_number=(min(pages) if pages else None),
            page_numbers=sorted(pages),
            chunk_order=chunk_order,
            project=ref_record.project,
            market=ref_record.market,
            product=ref_record.product,
            language=ref_record.language,
            source_record_ids=[r.record_id for r in source_records],
            metadata={
                **merged_record_meta,
                "source_entity_types": sorted({r.entity_type for r in source_records}),
                "num_source_records": len(source_records),
                "chunker": "semantic_chunker",
                "chunker_version": "0.2.0",
            },
        )

    def _make_table_chunks(
        self,
        start_order: int,
        record: EmbeddingRecord,
        fallback_section: Optional[str] = None,
    ) -> list[RagChunk]:
        """
        Делает один или несколько чанков из табличной записи.

        Если итоговый текст таблицы укладывается в max_chunk_chars — один
        чанк, как раньше. Если нет — режется на окна по границам строк
        таблицы, и в каждое окно повторяется «шапка» (название таблицы,
        колонки, секционный заголовок). Так гигантские финансовые таблицы
        и многостраничные сводки ВНИИПО перестают вылетать одним монстром
        на 50+ КБ, который bge-m3 всё равно обрежет.

        fallback_section — раздел из окружающего текста, используется
        если у самой записи таблицы нет section_path в metadata (что
        характерно для старых embedding_records, где exporter его не
        пробрасывал).
        """

        section_path = self._record_section(record) or fallback_section
        full_text = self._with_context_header(
            text=record.text,
            source_file=record.source_file,
            source_group=(record.metadata or {}).get("source_group"),
            section_path=section_path,
        )

        pages: list[int] = []
        if record.page_number is not None:
            pages = [int(record.page_number)]

        record_meta = record.metadata or {}
        base_meta = {
            **self._extract_document_level_metadata(record_meta),
            "section_path": section_path,
            "parser_name": record_meta.get("parser_name"),
            "parser_version": record_meta.get("parser_version"),
            "source_entity_types": [record.entity_type],
            "num_source_records": 1,
            "chunker": "semantic_chunker",
            "chunker_version": "0.3.0",
            "table_quality": record_meta.get("table_quality"),
            "columns": record_meta.get("columns"),
        }

        parts = self._split_table_text(full_text)
        total = len(parts)

        chunks: list[RagChunk] = []
        for offset, part_text in enumerate(parts):
            meta = dict(base_meta)
            if total > 1:
                meta["table_part"] = offset + 1
                meta["table_total_parts"] = total

            order = start_order + offset
            chunks.append(
                RagChunk(
                    chunk_id=f"{record.source_file}:rag_chunk:{order:05d}",
                    source_file=record.source_file,
                    source_type=record.source_type,
                    chunk_type="table",
                    text=part_text,
                    section_title=section_path,
                    page_number=(pages[0] if pages else None),
                    page_numbers=pages,
                    chunk_order=order,
                    project=record.project,
                    market=record.market,
                    product=record.product,
                    language=record.language,
                    source_record_ids=[record.record_id],
                    metadata=meta,
                )
            )

        return chunks

    def _split_table_text(self, text: str) -> list[str]:
        """
        Режет длинный текст таблицы на окна по границам строк.

        Поддерживает оба формата, которые приходят из exporter:
        - нормализованный («Таблица из документа: …\\nКолонки: …\\nДанные
          таблицы:\\n- Строка N: …»);
        - markdown pipe-таблица («| a | b |\\n| --- | --- |\\n| 1 | 2 |»).

        Если text укладывается в max_chunk_chars или формат не
        распознаётся — возвращает [text] без изменений.
        """

        if len(text) <= self.max_chunk_chars:
            return [text]

        lines = text.split("\n")

        # Ищем границу: где заканчивается «шапка» таблицы (prefix) и
        # начинаются «строки данных», которые и будем резать.
        data_start = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("- Строка"):
                data_start = i
                break
            # markdown pipe-таблица: строка-разделитель |---|---|
            if (
                stripped.startswith("|")
                and "---" in stripped
                and set(stripped.replace("|", "").strip()) <= set("-: ")
            ):
                data_start = i + 1
                break

        if data_start <= 0 or data_start >= len(lines):
            # Не нашли границу строк → не разрезать.
            return [text]

        prefix = "\n".join(lines[:data_start])
        data_lines = lines[data_start:]

        # Бюджет на строки данных в каждом окне (после повторения prefix).
        budget = self.max_chunk_chars - len(prefix) - 2
        if budget < 200:
            # Сама шапка таблицы слишком большая, дробить нечем.
            return [text]

        windows: list[str] = []
        current: list[str] = []
        current_size = 0

        for dl in data_lines:
            ln_size = len(dl) + 1  # +1 на перенос строки
            if current and current_size + ln_size > budget:
                windows.append(prefix + "\n" + "\n".join(current))
                current = []
                current_size = 0
            current.append(dl)
            current_size += ln_size

        if current:
            windows.append(prefix + "\n" + "\n".join(current))

        return windows or [text]

    def _make_visual_chunk(
        self,
        chunk_order: int,
        record: EmbeddingRecord,
    ) -> RagChunk:
        section_path = self._record_section(record)
        text = self._with_context_header(
            text=record.text,
            source_file=record.source_file,
            source_group=(record.metadata or {}).get("source_group"),
            section_path=section_path,
        )

        pages: list[int] = []
        if record.page_number is not None:
            pages = [int(record.page_number)]

        record_meta = record.metadata or {}

        return RagChunk(
            chunk_id=f"{record.source_file}:rag_chunk:{chunk_order:05d}",
            source_file=record.source_file,
            source_type=record.source_type,
            chunk_type="visual",
            text=text,
            section_title=section_path,
            page_number=(pages[0] if pages else None),
            page_numbers=pages,
            chunk_order=chunk_order,
            project=record.project,
            market=record.market,
            product=record.product,
            language=record.language,
            source_record_ids=[record.record_id],
            metadata={
                **self._extract_document_level_metadata(record_meta),
                "section_path": record_meta.get("section_path"),
                "parser_name": record_meta.get("parser_name"),
                "parser_version": record_meta.get("parser_version"),
                "source_entity_types": [record.entity_type],
                "num_source_records": 1,
                "chunker": "semantic_chunker",
                "chunker_version": "0.2.0",
                "image_path": record_meta.get("image_path"),
            },
        )

    # ============================================================
    # METADATA HELPERS
    # ============================================================

    @staticmethod
    def _extract_document_level_metadata(meta: dict) -> dict:
        """
        Берём только metadata, полезную на уровне документа/источника.

        Не тащим тяжёлые поля вроде normalized_rows.
        """

        keys = [
            "source_path",
            "source_group",
            "manifest_source_type",
            "topic",
            "selected_strategy",
            "parser_backend",
            "fallback_parser",
            "docling_failure",
            "docling_failure_before_repair",
            "docling_failure_after_repair",
            "pdf_repaired",
            "original_source_path",
            "repaired_source_path",
        ]

        return {key: meta.get(key) for key in keys if key in meta}

    def _merge_record_metadata(
        self,
        records: list[EmbeddingRecord],
    ) -> dict:
        """
        Собирает metadata для текстового чанка из нескольких records.

        Для document-level metadata берём первое ненулевое значение.
        section_path/parser_name/parser_version сохраняем отдельно.
        """

        merged: dict = {}

        for record in records:
            meta = record.metadata or {}

            for key, value in self._extract_document_level_metadata(meta).items():
                if key not in merged and value is not None:
                    merged[key] = value

        section_paths = []
        parser_names = set()
        parser_versions = set()

        for record in records:
            meta = record.metadata or {}

            section = meta.get("section_path")
            if section and section not in section_paths:
                section_paths.append(section)

            parser_name = meta.get("parser_name")
            if parser_name:
                parser_names.add(str(parser_name))

            parser_version = meta.get("parser_version")
            if parser_version:
                parser_versions.add(str(parser_version))

        if section_paths:
            merged["section_path"] = section_paths[0]
            if len(section_paths) > 1:
                merged["section_paths"] = section_paths

        if parser_names:
            merged["parser_names"] = sorted(parser_names)

        if parser_versions:
            merged["parser_versions"] = sorted(parser_versions)

        return merged

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _group_by_source(
        records: Iterable[EmbeddingRecord],
    ) -> Iterator[tuple[str, list[EmbeddingRecord]]]:
        current_key: Optional[str] = None
        bucket: list[EmbeddingRecord] = []

        for record in records:
            key = record.source_file
            if current_key is None:
                current_key = key

            if key != current_key:
                yield current_key, bucket
                current_key = key
                bucket = []

            bucket.append(record)

        if current_key is not None and bucket:
            yield current_key, bucket

    @staticmethod
    def _record_section(record: EmbeddingRecord) -> Optional[str]:
        meta = record.metadata or {}
        section = meta.get("section_path")
        if section:
            return str(section)
        return None

    @staticmethod
    def _join_parts(parts: list[str]) -> str:
        return "\n\n".join(p for p in parts if p and p.strip())

    def _with_context_header(
        self,
        text: str,
        source_file: Optional[str],
        source_group: Optional[str],
        section_path: Optional[str],
    ) -> str:
        """
        Богатый контекстный заголовок: [имя_файла | тема | Раздел: ...].

        Заголовок становится частью embedding'а, поэтому имя документа и
        тема попадают в вектор — поиск по «Сибкабель финансы» начнёт
        цеплять чанки, где в самом тексте только цифры.

        Если prepend_section_header выключен — возвращаем text как есть.
        Любая из составляющих может быть None: пропускаем её, но
        остальные оставляем.
        """

        if not self.prepend_section_header:
            return text

        parts: list[str] = []
        if source_file:
            parts.append(source_file)
        if source_group:
            parts.append(source_group)
        if section_path:
            parts.append(f"Раздел: {section_path}")

        if not parts:
            return text

        return f"[{' | '.join(parts)}]\n\n{text}"

    # ============================================================
    # TEXT NORMALIZATION
    # ============================================================

    _BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
    # «слово-\n далее» → «словодалее»: склейка переносов по дефису.
    # Срабатывает только когда после дефиса идёт строчная буква — то есть
    # это разрыв слова, а не дефисное составное вроде «социально-экономический».
    _HYPHEN_WRAP_RE = re.compile(r"(\w)-\s*\n\s*([a-zа-яё])", re.UNICODE)
    _MULTISPACE_RE = re.compile(r"[ \t]+")

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """
        Чистит текст перед чанкингом, чтобы embedding получал чистые
        токены, а не HTML-теги и обрывки слов.

        - <br> и <br/> → перенос строки;
        - «слово-\\n далее» → «словодалее»: склеиваем переносы по дефису;
        - схлопываем горизонтальные пробелы (не переносы строк);
        - убираем trailing whitespace по строкам.
        """

        if not text:
            return text

        text = cls._BR_RE.sub("\n", text)
        text = cls._HYPHEN_WRAP_RE.sub(r"\1\2", text)
        text = cls._MULTISPACE_RE.sub(" ", text)
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        return text.strip()

    # ============================================================
    # SENTENCE OVERLAP
    # ============================================================

    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?…])\s+(?=[A-ZА-ЯЁ0-9])")

    def _overlap_text(self, text: str) -> str:
        if self.overlap_sentences <= 0:
            return ""

        # Берём только последний абзац, чтобы не утаскивать заголовок секции.
        last_paragraph = text.rstrip().split("\n\n")[-1].strip()
        if not last_paragraph:
            return ""

        sentences = self._SENTENCE_SPLIT_RE.split(last_paragraph)
        if not sentences:
            return ""

        tail = sentences[-self.overlap_sentences:]
        return " ".join(s.strip() for s in tail if s.strip())