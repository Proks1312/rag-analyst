from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import (
    DoclingDocument,
    ListItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

from ingestion.models import (
    DocumentMetadata,
    ParsedBlock,
    ParsedDocument,
    ParsedTable,
)
from ingestion.parsers.base import BaseParser


class PdfDoclingParser(BaseParser):
    """
    PDF parser backend на базе Docling.

    Версия 0.3.0:
    - ходим по DoclingDocument через iterate_items() вместо markdown-конвертации;
    - получаем настоящие page_number через item.prov[0].page_no;
    - выделяем таблицы напрямую как TableItem -> ParsedTable;
    - section_title формируется через стек заголовков SectionHeaderItem.level
      и выглядит как полный путь "Раздел 3 / 3.2 Подраздел / 3.2.1 Подпункт";
    - игнорируем page_header / page_footer (повторяющийся мусор).

    Картинки (PictureItem) на этом этапе не превращаем в ParsedBlock — ими
    занимается отдельный PdfPageRenderer, который рендерит страницы и
    кладёт их в document.visuals.
    """

    parser_name: str = "docling"
    parser_version: str = "0.3.0"
    supported_extensions: set[str] = {".pdf"}

    # Метки, которые мы хотим выкинуть полностью (повторяющиеся колонтитулы).
    _SKIP_LABELS: set[str] = {
        "page_header",
        "page_footer",
        "footnote",
    }

    def __init__(self) -> None:
        self.converter = DocumentConverter()

    # ============================================================
    # PUBLIC
    # ============================================================

    def parse(
        self,
        path: Path,
        metadata: DocumentMetadata,
    ) -> ParsedDocument:
        metadata.parser_name = self.parser_name
        metadata.parser_version = self.parser_version

        dl_doc = self._convert_with_repair(path=path, metadata=metadata)

        blocks, tables = self._walk_document(dl_doc=dl_doc, metadata=metadata)

        return ParsedDocument(
            metadata=metadata,
            blocks=blocks,
            tables=tables,
        )

    # ============================================================
    # CONVERT + AUTO-REPAIR
    # ============================================================

    def _convert_with_repair(
        self,
        path: Path,
        metadata: DocumentMetadata,
    ) -> DoclingDocument:
        """
        Конвертирует PDF через Docling.

        Многие PDF (например, выгрузки из Контур.Фокуса) падают с
        ConversionError / "Inconsistent number of pages" из-за «битого»
        дерева страниц. Такие PDF чинятся пересохранением через PyMuPDF
        (garbage collection + clean). Поэтому при первой ошибке мы
        пробуем починить документ и сконвертировать ещё раз.

        Если и после repair не получилось — пробрасываем исключение,
        чтобы сработал внешний fallback (pipeline / HybridParser → PyMuPDF).
        """

        try:
            result = self.converter.convert(path)
            return result.document
        except Exception as exc:
            first_error = f"{type(exc).__name__}: {exc}"

            # Записываем первую ошибку сразу, до repair и повторной
            # конвертации — чтобы она не потерялась, если они тоже упадут.
            metadata.extra["docling_first_error"] = first_error

            repaired_path = self._repair_pdf(path)

            if repaired_path is None:
                raise

            print(
                f"[docling] convert failed ({first_error}); "
                f"retrying with repaired PDF"
            )

            try:
                result = self.converter.convert(repaired_path)
            finally:
                try:
                    repaired_path.unlink()
                except Exception:
                    pass

            metadata.extra["docling_used_repaired_pdf"] = True

            return result.document

    @staticmethod
    def _repair_pdf(path: Path) -> Optional[Path]:
        """
        Чинит PDF пересборкой только валидных, непустых страниц
        в новый чистый документ.

        Зачем сложнее, чем простой save(garbage=4, clean=True):
        некоторые PDF (выгрузки из iLovePDF / Контур.Фокуса) имеют
        раздутый /Pages — например, /Count говорит 114, а реально видны
        только 20. Простой save с garbage сохраняет ВСЕ 114, включая
        битые/пустые «фантомы». Docling потом OOM-ится, гоня layout-model
        по фантомным страницам.

        Логика:
          1. Перебираем страницы оригинала.
          2. Каждую открываем индивидуально через try/except — битые скипаем.
          3. У страницы должен быть хотя бы какой-то контент:
             текст, графика или картинка. Иначе скип.
          4. Валидные страницы поштучно копируем в новый документ через
             insert_pdf(); это переносит content stream и ресурсы,
             но НЕ тащит мёртвые ссылки из /Pages.
          5. Сохраняем новый документ с garbage=4 deflate clean.

        Возвращает путь к временному файлу или None, если починка
        не удалась или не осталось ни одной валидной страницы.
        """

        try:
            import fitz  # PyMuPDF

            orig = fitz.open(path)
        except Exception:
            return None

        try:
            new_doc = fitz.open()  # пустой целевой
            kept = 0
            for i in range(orig.page_count):
                try:
                    page = orig[i]
                except Exception:
                    continue

                # Считаем страницу валидной, если на ней хоть что-то есть.
                try:
                    has_text = bool(page.get_text("text").strip())
                except Exception:
                    has_text = False
                try:
                    has_images = bool(page.get_images(full=True))
                except Exception:
                    has_images = False
                try:
                    has_drawings = bool(page.get_drawings())
                except Exception:
                    has_drawings = False

                if not (has_text or has_images or has_drawings):
                    continue

                try:
                    new_doc.insert_pdf(orig, from_page=i, to_page=i)
                    kept += 1
                except Exception:
                    continue

            if kept == 0:
                new_doc.close()
                return None

            fd, tmp_name = tempfile.mkstemp(suffix=".pdf", prefix="docling_repair_")
            os.close(fd)
            tmp_path = Path(tmp_name)

            try:
                new_doc.save(
                    str(tmp_path),
                    garbage=4,
                    deflate=True,
                    clean=True,
                )
            finally:
                new_doc.close()

            print(
                f"[docling] repair: kept {kept}/{orig.page_count} pages "
                f"(dropped {orig.page_count - kept} blank/invalid)"
            )

            return tmp_path
        finally:
            orig.close()

    # ============================================================
    # MAIN WALK
    # ============================================================

    def _walk_document(
        self,
        dl_doc: DoclingDocument,
        metadata: DocumentMetadata,
    ) -> tuple[list[ParsedBlock], list[ParsedTable]]:
        blocks: list[ParsedBlock] = []
        tables: list[ParsedTable] = []

        heading_stack: list[tuple[int, str]] = []
        block_order = 0
        table_order = 0

        for item, _depth in self._iterate_items_safe(dl_doc):
            # 1) Document title — становится корнем иерархии.
            if isinstance(item, TitleItem):
                text = self._safe_text(item)
                if not text:
                    continue

                heading_stack = [(0, text)]

                block = self._mk_block(
                    text=text,
                    block_type="title",
                    block_order=block_order,
                    metadata=metadata,
                    page_number=self._safe_page_no(item),
                    section_path=text,
                )
                blocks.append(block)
                block_order += 1
                continue

            # 2) Section heading — обновляем стек, сам блок не сохраняем
            #    как paragraph (он живёт в section_path).
            if isinstance(item, SectionHeaderItem):
                text = self._safe_text(item)
                if not text:
                    continue

                level = max(1, int(getattr(item, "level", 1)))

                # Срезаем стек до уровня выше текущего и кладём новый.
                heading_stack = [(l, t) for (l, t) in heading_stack if l < level]
                heading_stack.append((level, text))
                continue

            # 3) Table — отдельной сущностью, не в blocks.
            if isinstance(item, TableItem):
                table = self._make_table(
                    table_item=item,
                    dl_doc=dl_doc,
                    metadata=metadata,
                    table_order=table_order,
                    section_path=self._compose_section_path(heading_stack),
                )
                if table is not None:
                    tables.append(table)
                    table_order += 1
                continue

            # 4) ListItem — отдельный block_type="list".
            if isinstance(item, ListItem):
                text = self._safe_text(item)
                if not text:
                    continue

                marker = getattr(item, "marker", "-") or "-"
                rendered = f"{marker} {text}".strip()

                block = self._mk_block(
                    text=rendered,
                    block_type="list",
                    block_order=block_order,
                    metadata=metadata,
                    page_number=self._safe_page_no(item),
                    section_path=self._compose_section_path(heading_stack),
                )
                blocks.append(block)
                block_order += 1
                continue

            # 5) TextItem — основной контент (paragraph / text / caption / ...).
            if isinstance(item, TextItem):
                label_name = self._label_name(item)
                if label_name in self._SKIP_LABELS:
                    continue

                text = self._safe_text(item)
                if not text:
                    continue

                block_type = "paragraph"
                if label_name == "caption":
                    block_type = "note"

                block = self._mk_block(
                    text=text,
                    block_type=block_type,
                    block_order=block_order,
                    metadata=metadata,
                    page_number=self._safe_page_no(item),
                    section_path=self._compose_section_path(heading_stack),
                )
                blocks.append(block)
                block_order += 1
                continue

            # 6) PictureItem — пропускаем, ими занимается PdfPageRenderer.
            if isinstance(item, PictureItem):
                continue

            # Прочие типы (GroupItem, RootItem и т.п.) — игнор.

        return blocks, tables

    # ============================================================
    # ITEM HELPERS
    # ============================================================

    @staticmethod
    def _iterate_items_safe(dl_doc: DoclingDocument) -> Iterable[tuple[object, int]]:
        """
        Обёртка вокруг iterate_items, чтобы не падать на edge-cases
        (например, документ без body или с неконсистентной структурой).
        """

        try:
            yield from dl_doc.iterate_items()
        except Exception as exc:
            raise RuntimeError(
                f"DoclingDocument.iterate_items() failed: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _safe_text(item: object) -> str:
        text = getattr(item, "text", None)
        if not text:
            return ""
        return str(text).strip()

    @staticmethod
    def _safe_page_no(item: object) -> Optional[int]:
        """
        Достаём номер страницы из item.prov[0].page_no.

        item.prov может быть пустым (например, у корневых элементов).
        page_no в Docling 1-indexed.
        """

        try:
            prov = getattr(item, "prov", None) or []
            if not prov:
                return None
            page_no = getattr(prov[0], "page_no", None)
            if page_no is None:
                return None
            return int(page_no)
        except Exception:
            return None

    @staticmethod
    def _label_name(item: object) -> str:
        """
        Возвращает имя ярлыка элемента (например, 'paragraph', 'page_header').
        """

        label = getattr(item, "label", None)
        if label is None:
            return ""
        # DocItemLabel — это Enum, у него есть .value.
        value = getattr(label, "value", label)
        return str(value).lower()

    @staticmethod
    def _compose_section_path(
        heading_stack: list[tuple[int, str]],
    ) -> Optional[str]:
        if not heading_stack:
            return None
        parts = [t for _, t in heading_stack if t]
        if not parts:
            return None
        return " / ".join(parts)

    def _mk_block(
        self,
        text: str,
        block_type: str,
        block_order: int,
        metadata: DocumentMetadata,
        page_number: Optional[int],
        section_path: Optional[str],
    ) -> ParsedBlock:
        return ParsedBlock(
            block_text=text,
            block_type=block_type,
            block_order=block_order,
            source_file=metadata.source_file,
            source_type=metadata.source_type,
            page_number=page_number,
            section_title=section_path,
            language=metadata.language,
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            metadata={
                "project": metadata.project,
                "market": metadata.market,
                "product": metadata.product,
                "parser_backend": self.parser_name,
                "section_path": section_path,
            },
        )

    # ============================================================
    # TABLES
    # ============================================================

    def _make_table(
        self,
        table_item: TableItem,
        dl_doc: DoclingDocument,
        metadata: DocumentMetadata,
        table_order: int,
        section_path: Optional[str],
    ) -> Optional[ParsedTable]:
        page_no = self._safe_page_no(table_item)

        # Caption taблицы — например, "Таблица 1.1. Динамика ..."
        table_title: Optional[str] = None
        try:
            cap = table_item.caption_text(dl_doc)
            if cap and cap.strip():
                table_title = cap.strip()
        except Exception:
            table_title = None

        # Markdown-представление таблицы.
        markdown_table = ""
        try:
            markdown_table = table_item.export_to_markdown(doc=dl_doc) or ""
        except Exception:
            markdown_table = ""

        # Raw data — список списков. Берём из DataFrame.
        raw_data: list[list] = []
        try:
            df = table_item.export_to_dataframe(doc=dl_doc)
            if df is not None and not df.empty:
                raw_data = [list(df.columns)] + df.astype(str).values.tolist()
        except Exception:
            raw_data = []

        # Если ничего не получили — пропускаем.
        if not markdown_table.strip() and not raw_data:
            return None

        return ParsedTable(
            markdown_table=markdown_table,
            raw_data=raw_data,
            table_order=table_order,
            source_file=metadata.source_file,
            source_type=metadata.source_type,
            page_number=page_no,
            table_title=table_title or section_path,
            language=metadata.language,
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            metadata={
                "project": metadata.project,
                "market": metadata.market,
                "product": metadata.product,
                "parser_backend": self.parser_name,
                "section_path": section_path,
                "table_format": "markdown",
                "from_native_docling_table": True,
            },
        )
