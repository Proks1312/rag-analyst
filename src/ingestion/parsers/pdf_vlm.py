"""
VLM-based PDF parser.

«Умный» парсер последнего рубежа: рендерит страницу PDF в картинку и
отдаёт её локальной vision-модели через Ollama (по умолчанию
qwen3-vl:8b). VLM «читает» страницу как человек и возвращает markdown,
который мы разбираем на блоки текста и таблицы.

Зачем нужен:
- сканы, фотографии страниц, кривая вёрстка;
- сложные финансовые/китайские таблицы, которые геометрия не вытягивает;
- общий «умный» fallback, когда Docling/PyMuPDF дают мусор.

Как встаёт в существующую архитектуру:
- parser_name = "vlm";
- занимает слот стратегии `ocr_required` — её и так выбирает
  профилировщик для документов без текстового слоя;
- vlm-парсер вызывается явно через --parser vlm (или ocr_required).

Зависимости:
- Ollama ≥ 0.12.7, поднятая локально;
- модель скачана: `ollama pull qwen3-vl:8b`;
- PyMuPDF (fitz) и requests — оба уже в проекте;
- импортируются лениво внутри parse(), чтобы отсутствие пакета не
  ломало запуск пайплайна для PDF/Excel.

Конфигурация через .env:
    OLLAMA_URL        URL генерации Ollama (тот же, что у answer_question.py)
    OLLAMA_VLM_MODEL  имя vision-модели, по умолчанию qwen3-vl:8b
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Optional

from ..models import DocumentMetadata, ParsedBlock, ParsedDocument, ParsedTable
from .base import BaseParser


class VlmPdfParser(BaseParser):
    parser_name: str = "vlm"
    parser_version: str = "0.1.0"
    supported_extensions: set[str] = {".pdf"}

    DEFAULT_MODEL: str = "qwen3-vl:8b"
    DEFAULT_OLLAMA_URL: str = "http://localhost:11434/api/generate"
    DEFAULT_DPI: int = 150
    DEFAULT_TIMEOUT: int = 240  # секунд на страницу

    PROMPT: str = (
        "You are a document-parsing assistant. Convert the content of this PDF "
        "page image into clean markdown.\n\n"
        "Rules:\n"
        "- Preserve the document's original language EXACTLY (Russian, English, "
        "Chinese, etc.). Do NOT translate.\n"
        "- Headings: use # ONLY for a real document title or major section "
        "heading. Use ## for genuine subsections. Do NOT use #/##/### for "
        "short labels, captions, venue/organization names, dates, or every "
        "prominent line on a cover/title page — those are PARAGRAPHS. If a "
        "page has one main title and several smaller pieces of info, output "
        "one # for the title and treat the rest as plain paragraphs.\n"
        "- Render every table as a proper markdown pipe table. Keep all "
        "numbers exactly as printed. Preserve column alignment.\n"
        "- Preserve list structure with - or numbered items.\n"
        "- Skip page numbers, running headers, footers and other page chrome.\n"
        "- For a chart or figure without machine-readable text, write a single "
        "line like: > Figure: <short description>\n"
        "- Output ONLY the markdown content. No commentary, no code fences "
        "around the whole output."
    )

    def __init__(
        self,
        model: Optional[str] = None,
        ollama_url: Optional[str] = None,
        dpi: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_VLM_MODEL") or self.DEFAULT_MODEL
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL") or self.DEFAULT_OLLAMA_URL
        self.dpi = int(dpi or os.getenv("VLM_DPI") or self.DEFAULT_DPI)
        self.timeout = int(timeout or os.getenv("VLM_PAGE_TIMEOUT") or self.DEFAULT_TIMEOUT)

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

        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise RuntimeError(
                "VlmPdfParser требует PyMuPDF. Установите: pip install pymupdf"
            ) from exc

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "VlmPdfParser требует requests. Установите: pip install requests"
            ) from exc

        blocks: list[ParsedBlock] = []
        tables: list[ParsedTable] = []
        block_order = 0
        table_order = 0
        current_section: Optional[str] = None

        doc = fitz.open(path)
        try:
            page_count = doc.page_count
            print(
                f"[vlm] {path.name}: {page_count} страниц "
                f"-> model={self.model}, dpi={self.dpi}"
            )

            for page_index in range(page_count):
                page_number = page_index + 1
                try:
                    page = doc.load_page(page_index)
                    image_bytes = self._render_page_png(page)
                    markdown = self._call_vlm(image_bytes, requests_mod=requests)
                except Exception as exc:
                    print(
                        f"[vlm] страница {page_number} упала: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    # Если упала первая же страница — это явно конфигурация
                    # (Ollama не доступна / модель не скачана). Падаем громко,
                    # чтобы pipeline увидел и обработал.
                    if page_number == 1 and not blocks and not tables:
                        raise
                    continue

                # Разбираем markdown страницы на сегменты и эмитим
                # ParsedBlock / ParsedTable.
                for kind, payload in self._split_markdown(markdown):
                    if kind == "heading":
                        text, level = payload
                        # Только заголовки 1-2 уровня меняют section_path.
                        # Глубже (###+) — это, как правило, ярлыки/подписи
                        # на cover-страницах, и сбрасывать на них раздел
                        # значит дробить документ на крошечные чанки.
                        if level <= 2:
                            current_section = text
                        blocks.append(self._mk_block(
                            text=text,
                            block_type="heading",
                            block_order=block_order,
                            metadata=metadata,
                            page_number=page_number,
                            section_path=current_section,
                        ))
                        block_order += 1
                    elif kind == "table":
                        tab = self._mk_table(
                            markdown_table=payload,
                            table_order=table_order,
                            page_number=page_number,
                            metadata=metadata,
                            section_path=current_section,
                        )
                        if tab is not None:
                            tables.append(tab)
                            table_order += 1
                    elif kind == "list":
                        blocks.append(self._mk_block(
                            text=payload,
                            block_type="list",
                            block_order=block_order,
                            metadata=metadata,
                            page_number=page_number,
                            section_path=current_section,
                        ))
                        block_order += 1
                    elif kind == "paragraph":
                        blocks.append(self._mk_block(
                            text=payload,
                            block_type="paragraph",
                            block_order=block_order,
                            metadata=metadata,
                            page_number=page_number,
                            section_path=current_section,
                        ))
                        block_order += 1
        finally:
            doc.close()

        return ParsedDocument(
            metadata=metadata,
            blocks=blocks,
            tables=tables,
            visuals=[],
        )

    # ============================================================
    # RENDER + VLM CALL
    # ============================================================

    def _render_page_png(self, page) -> bytes:
        """Рендерит страницу PDF в PNG. DPI настраивается."""
        pix = page.get_pixmap(dpi=self.dpi)
        return pix.tobytes("png")

    def _call_vlm(self, image_bytes: bytes, requests_mod) -> str:
        """
        Зовёт Ollama VLM в стриминговом режиме.

        Стриминг важен, чтобы self.timeout считался не «весь ответ за N
        секунд», а «нет ни одного токена за N секунд». На сложных страницах
        VLM генерит много markdown-токенов, и блокирующий запрос упирался
        в таймаут, хотя модель честно работала. Со стримом такая страница
        может молотиться сколько надо — лишь бы поток не вставал, — а
        реальный ступор Ollama мы всё равно поймаем.
        """

        b64 = base64.b64encode(image_bytes).decode("ascii")

        payload = {
            "model": self.model,
            "prompt": self.PROMPT,
            "images": [b64],
            "stream": True,
            "options": {
                "temperature": 0.1,
                "num_ctx": 8192,
            },
        }

        session = requests_mod.Session()
        session.trust_env = False  # тот же паттерн, что в answer_question.py

        response = session.post(
            self.ollama_url,
            json=payload,
            timeout=self.timeout,
            stream=True,
        )

        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama вернула 404: модель {self.model!r} не найдена. "
                f"Скачайте её: ollama pull {self.model}"
            )

        response.raise_for_status()

        # Ollama в stream-режиме отдаёт NDJSON: по одной JSON-строке на
        # каждый чанк токенов, плюс финальный с "done": true.
        parts: list[str] = []
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            piece = chunk.get("response")
            if piece:
                parts.append(piece)
            if chunk.get("done"):
                break

        text = "".join(parts).strip()
        return self._strip_outer_fence(text)

    @staticmethod
    def _strip_outer_fence(text: str) -> str:
        """
        VLM-ки иногда оборачивают весь ответ в ```markdown ... ```, даже
        когда инструкция просит этого не делать. Срезаем такую обёртку.
        """
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    # ============================================================
    # MARKDOWN -> SEGMENTS
    # ============================================================

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
    _LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.\)])\s+")
    _SEPARATOR_RE = re.compile(r"^:?-+:?$")

    def _split_markdown(self, md: str) -> list[tuple[str, object]]:
        """
        Режет markdown страницы на сегменты:
          ("heading", (text, level))
          ("table",   markdown_text)
          ("list",    markdown_text)
          ("paragraph", markdown_text)
        """
        lines = md.splitlines()
        segments: list[tuple[str, object]] = []
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                i += 1
                continue

            # 1) Heading
            m = self._HEADING_RE.match(stripped)
            if m:
                level = len(m.group(1))
                text = m.group(2).strip()
                if text:
                    segments.append(("heading", (text, level)))
                i += 1
                continue

            # 2) Table — несколько строк подряд, начинающихся с '|'
            if stripped.startswith("|"):
                table_lines: list[str] = []
                while i < n and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i].rstrip())
                    i += 1
                # Таблица должна быть хотя бы 2 строки (шапка + разделитель/строка).
                if len(table_lines) >= 2:
                    segments.append(("table", "\n".join(table_lines)))
                continue

            # 3) List
            if self._LIST_RE.match(line):
                list_lines: list[str] = []
                while i < n:
                    ln = lines[i]
                    s = ln.strip()
                    if not s:
                        break
                    # либо новый list-маркер, либо продолжение пункта
                    # (отступ + текст).
                    if self._LIST_RE.match(ln) or (ln.startswith((" ", "\t")) and s):
                        list_lines.append(ln.rstrip())
                        i += 1
                        continue
                    break
                if list_lines:
                    segments.append(("list", "\n".join(list_lines)))
                continue

            # 4) Paragraph — обычные строки до следующего спец-сегмента
            para_lines: list[str] = []
            while i < n:
                ln = lines[i]
                s = ln.strip()
                if not s:
                    break
                if s.startswith("#") or s.startswith("|"):
                    break
                if self._LIST_RE.match(ln):
                    break
                para_lines.append(ln.rstrip())
                i += 1
            text = "\n".join(para_lines).strip()
            if text:
                segments.append(("paragraph", text))

        return segments

    # ============================================================
    # FACTORIES
    # ============================================================

    def _mk_block(
        self,
        text: str,
        block_type: str,
        block_order: int,
        metadata: DocumentMetadata,
        page_number: int,
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
                "vlm_model": self.model,
            },
        )

    def _mk_table(
        self,
        markdown_table: str,
        table_order: int,
        page_number: int,
        metadata: DocumentMetadata,
        section_path: Optional[str],
    ) -> Optional[ParsedTable]:
        if not markdown_table.strip():
            return None

        # raw_data — распарсим markdown table в list[list[str]], выкинув
        # строку-разделитель |---|---|.
        raw_data: list[list[str]] = []
        for ln in markdown_table.splitlines():
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            non_empty = [c for c in cells if c]
            if non_empty and all(self._SEPARATOR_RE.match(c) for c in non_empty):
                continue  # пропускаем строку-разделитель
            raw_data.append(cells)

        if not raw_data:
            return None

        return ParsedTable(
            markdown_table=markdown_table,
            raw_data=raw_data,
            table_order=table_order,
            source_file=metadata.source_file,
            source_type=metadata.source_type,
            page_number=page_number,
            table_title=section_path,
            language=metadata.language,
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            metadata={
                "project": metadata.project,
                "market": metadata.market,
                "product": metadata.product,
                "parser_backend": self.parser_name,
                "parser_name": self.parser_name,
                "parser_version": self.parser_version,
                "section_path": section_path,
                "table_format": "markdown",
                "from_vlm": True,
                "vlm_model": self.model,
            },
        )
