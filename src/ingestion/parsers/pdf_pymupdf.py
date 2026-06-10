from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import fitz  # PyMuPDF

from ..models import DocumentMetadata, ParsedBlock, ParsedDocument, ParsedTable
from .base import BaseParser


class PdfPyMuPdfParser(BaseParser):
    """
    Fallback PDF parser на PyMuPDF.

    Используется для PDF, которые Docling не может обработать:
    - нестандартная структура PDF;
    - Inconsistent number of pages;
    - std::bad_alloc на больших документах.

    Особенности:
    - читает текстовый слой PDF постранично;
    - сохраняет page_number;
    - аккуратнее выделяет заголовки;
    - не превращает табличные строки в section_title;
    - большие табличные страницы режет на page/table-like chunks;
    - v0.3.0: извлекает таблицы через нативный page.find_tables()
      и пробрасывает section_title в текстовые блоки;
    - v0.4.0: мульти-стратегия find_tables() — на каждой странице
      пробует несколько режимов и выбирает таблицы с лучшим разбиением
      по колонкам (важно для финансовой отчётности без линий-границ).
    """

    parser_name: str = "pymupdf"
    parser_version: str = "0.4.0"
    supported_extensions: set[str] = {".pdf"}

    # Стратегии find_tables() в порядке приоритета.
    # Каждая — (vertical_strategy, horizontal_strategy).
    # На каждой странице пробуем по очереди и выбираем набор таблиц
    # с лучшим score (см. _score_raw_table). Ранний выход — если
    # стратегия дала «хорошие» многоколоночные таблицы.
    #
    # Почему именно так:
    # - "lines_strict" / "lines" — для таблиц с настоящими линиями-границами;
    # - "text" по вертикали — для финансовой отчётности (Контур.Фокус и т.п.),
    #   где колонки разделены не линиями, а выравниванием чисел.
    DEFAULT_TABLE_STRATEGIES: list[tuple[str, str]] = [
        ("lines_strict", "lines_strict"),
        ("text", "lines"),
        ("text", "text"),
    ]

    # Таблица считается «хорошей», если у неё хотя бы столько колонок.
    good_table_min_cols: int = 4

    def __init__(
        self,
        extract_tables: bool = True,
        table_strategies: list[tuple[str, str]] | None = None,
    ) -> None:
        """
        :param extract_tables: включить извлечение таблиц через find_tables().
        :param table_strategies: список стратегий (vertical, horizontal) для
            find_tables(). На каждой странице пробуем по очереди и выбираем
            набор с наибольшим числом колонок / заполненностью. Если None —
            используется DEFAULT_TABLE_STRATEGIES.
        """
        self.extract_tables = extract_tables
        self.table_strategies = table_strategies or list(self.DEFAULT_TABLE_STRATEGIES)

    def parse(
        self,
        path: Path,
        metadata: DocumentMetadata,
    ) -> ParsedDocument:
        blocks: list[ParsedBlock] = []
        tables: list[ParsedTable] = []

        doc = fitz.open(path)
        block_order = 0
        table_order = 0

        # Универсально находим колонтитулы: строки, которые повторяются
        # на многих страницах документа. Без хардкода под конкретный отчёт.
        repeated_lines = self._collect_repeated_lines(doc)

        # current_section — последний осмысленный heading.
        # Пробрасываем его в section_title последующих текстовых блоков,
        # чтобы semantic chunker мог группировать по разделам.
        current_section: str | None = None

        try:
            import time as _t
            total_pages = doc.page_count
            t_doc_start = _t.time()

            for page_index in range(total_pages):
                page_number = page_index + 1
                page = doc.load_page(page_index)

                t_page_start = _t.time()

                # 1) Таблицы через нативный find_tables().
                # current_section на этот момент = последний заголовок,
                # увиденный до текущей страницы; используется как
                # section_path в metadata таблиц.
                table_rects: list = []
                if self.extract_tables:
                    page_tables, table_rects = self._extract_page_tables(
                        page=page,
                        page_number=page_number,
                        metadata=metadata,
                        start_order=table_order,
                        current_section=current_section,
                    )
                    tables.extend(page_tables)
                    table_order += len(page_tables)

                t_tables = _t.time() - t_page_start

                # 2) Текст страницы, исключая зоны таблиц.
                raw_text = self._page_text_excluding_tables(page, table_rects)
                lines = self._clean_lines(raw_text)

                if not lines:
                    continue

                page_blocks = self._build_page_blocks(
                    lines=lines,
                    page_number=page_number,
                    repeated_lines=repeated_lines,
                )

                for item in page_blocks:
                    block_text = item["text"].strip()
                    block_type = item["type"]

                    if not block_text:
                        continue

                    # Heading обновляет current_section.
                    if block_type == "heading":
                        candidate = self._clean_section_candidate(block_text)
                        if candidate:
                            current_section = candidate

                    blocks.append(
                        ParsedBlock(
                            block_text=block_text,
                            block_type=block_type,
                            block_order=block_order,
                            source_file=metadata.source_file,
                            source_type=metadata.source_type,
                            page_number=page_number,
                            section_title=current_section,
                            language=metadata.language,
                            parser_name=self.parser_name,
                            parser_version=self.parser_version,
                            metadata={
                                **(metadata.extra or {}),
                                "project": metadata.project,
                                "market": metadata.market,
                                "product": metadata.product,
                                "parser_backend": self.parser_name,
                                "fallback_parser": True,
                                "page_index": page_index,
                                "page_number": page_number,
                                "block_strategy": item.get("strategy"),
                                "section_path": current_section,
                            },
                        )
                    )

                    block_order += 1

                t_total = _t.time() - t_page_start
                # Печать прогресса: на каждой 5-й странице или если страница долгая.
                if page_number % 5 == 0 or t_total > 2.0 or page_number == total_pages:
                    print(
                        f"[pymupdf] page {page_number}/{total_pages}: "
                        f"tables={len(table_rects)} t_tables={t_tables:.2f}s "
                        f"t_total={t_total:.2f}s",
                        flush=True,
                    )

            print(
                f"[pymupdf] done {total_pages} pages in {_t.time() - t_doc_start:.1f}s",
                flush=True,
            )

        finally:
            doc.close()

        return ParsedDocument(
            metadata=self._metadata_with_parser(metadata),
            blocks=blocks,
            tables=tables,
            visuals=[],
        )

    # ============================================================
    # TABLE EXTRACTION (find_tables)
    # ============================================================

    def _extract_page_tables(
        self,
        page,
        page_number: int,
        metadata: DocumentMetadata,
        start_order: int,
        current_section: str | None = None,
    ) -> tuple[list[ParsedTable], list]:
        """
        Извлекает таблицы со страницы через PyMuPDF find_tables().

        Пробует несколько стратегий (см. table_strategies) и выбирает
        набор таблиц с лучшим суммарным score. Это критично для финансовой
        отчётности: lines_strict часто склеивает все колонки в 1-2, а
        vertical_strategy="text" корректно делит их по выравниванию чисел.

        Для каждой таблицы пытается найти caption над её bbox — строку
        вида «Таблица N.» или «Таблица N. Название», лежащую непосредственно
        выше первой строки таблицы. Caption становится table_title.

        current_section прокидывается в metadata.section_path таблицы,
        чтобы semantic chunker и эмбеддинги знали, к какому разделу
        документа эта таблица относится.

        Возвращает (список ParsedTable, список fitz.Rect табличных зон).
        Табличные зоны нужны, чтобы вырезать табличный текст из обычных
        блоков и не дублировать его.
        """

        best_raw: list[dict] = []
        best_score = -1.0
        best_strategy: tuple[str, str] | None = None

        for v_strat, h_strat in self.table_strategies:
            candidate = self._find_tables_raw(page, v_strat, h_strat)

            score = sum(item["score"] for item in candidate)

            if score > best_score:
                best_score = score
                best_raw = candidate
                best_strategy = (v_strat, h_strat)

            # Ранний выход: стратегия уже дала «хорошие» многоколоночные
            # таблицы — нет смысла тратить время на остальные.
            if candidate and all(
                item["ncols"] >= self.good_table_min_cols for item in candidate
            ):
                break

        if not best_raw:
            return [], []

        # Кандидаты на caption — все текстовые блоки страницы.
        # Используем один раз для всех таблиц на странице, чтобы не дёргать
        # page.get_text("blocks") по нескольку раз.
        page_text_blocks = self._collect_page_text_blocks(page)

        parsed_tables: list[ParsedTable] = []
        table_rects: list = []

        for item in best_raw:
            bbox = item.get("bbox")
            rect = None
            if bbox is not None:
                try:
                    rect = fitz.Rect(bbox)
                    table_rects.append(rect)
                except Exception:
                    rect = None

            caption = self._find_table_caption_above(
                rect=rect,
                page_text_blocks=page_text_blocks,
            )

            parsed_tables.append(
                ParsedTable(
                    markdown_table=item["markdown"],
                    raw_data=item["rows"],
                    table_order=start_order + len(parsed_tables),
                    source_file=metadata.source_file,
                    source_type=metadata.source_type,
                    page_number=page_number,
                    table_title=caption or current_section,
                    language=metadata.language,
                    parser_name=self.parser_name,
                    parser_version=self.parser_version,
                    metadata={
                        "project": metadata.project,
                        "market": metadata.market,
                        "product": metadata.product,
                        "parser_backend": self.parser_name,
                        "table_format": "markdown",
                        "from_pymupdf_find_tables": True,
                        "table_strategy": (
                            f"{best_strategy[0]}/{best_strategy[1]}"
                            if best_strategy
                            else None
                        ),
                        "table_n_cols": item["ncols"],
                        "page_number": page_number,
                        "section_path": current_section,
                        "table_caption": caption,
                    },
                )
            )

        return parsed_tables, table_rects

    @staticmethod
    def _collect_page_text_blocks(page) -> list[tuple[float, float, float, float, str]]:
        """
        Возвращает все текстовые блоки страницы как
        (x0, y0, x1, y1, text). Используется для поиска caption над таблицей.
        """

        try:
            raw = page.get_text("blocks") or []
        except Exception:
            return []

        result: list[tuple[float, float, float, float, str]] = []
        for b in raw:
            # PyMuPDF возвращает (x0, y0, x1, y1, text, block_no, block_type)
            if len(b) < 5:
                continue
            try:
                x0, y0, x1, y1, text = float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4])
            except Exception:
                continue
            text = text.strip()
            if text:
                result.append((x0, y0, x1, y1, text))
        return result

    def _find_table_caption_above(
        self,
        rect,
        page_text_blocks: list[tuple[float, float, float, float, str]],
        max_gap_pt: float = 120.0,
    ) -> str | None:
        """
        Ищет caption над bbox таблицы.

        В русских статсборниках caption бывает оформлен по-разному:
          • прижатая к правому краю строка «Таблица N» отдельным блоком,
            а ниже — центрированное название таблицы;
          • один блок «Таблица N\\nНазвание таблицы»;
          • один блок «Section heading\\nТаблица N» — caption склеен
            с заголовком раздела.

        Поэтому:
          1. Берём все блоки выше bbox таблицы (y1 <= rect.y0)
             в пределах max_gap_pt.
          2. Для каждого сканируем ВСЕ строки на матч «Таблица N».
          3. Если матч найден — возвращаем эту строку, при возможности
             склеивая со следующей не-пустой строкой того же блока
             (она обычно — описание таблицы).
          4. Дополнительно: если выше bbox таблицы лежит маленький блок
             только из «Таблица N» — пробуем подцепить ближайший
             центрированный блок над bbox таблицы как описание.

        Возвращает None, если caption не найден.
        """

        if rect is None or not page_text_blocks:
            return None

        try:
            top = float(rect.y0)
        except Exception:
            return None

        candidates: list[tuple[float, float, float, float, str]] = []
        for x0, y0, x1, y1, text in page_text_blocks:
            if y1 > top:
                continue
            if top - y1 > max_gap_pt:
                continue
            candidates.append((x0, y0, x1, y1, text))

        if not candidates:
            return None

        # Самые близкие сверху первые.
        candidates.sort(key=lambda b: b[3], reverse=True)

        # Шаг 1: ищем блок, в котором есть строка «Таблица N».
        for x0, y0, x1, y1, text in candidates:
            lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
            for idx, line in enumerate(lines):
                if not self._is_table_title(line):
                    continue

                # Нашли «Таблица N» — попробуем склеить с описанием.
                caption = line

                # 1) Описание в этом же блоке после «Таблица N»:
                if idx + 1 < len(lines):
                    next_line = lines[idx + 1]
                    # пропускаем строки, которые выглядят как section heading
                    # или сами «Таблица N»
                    if (
                        not self._is_table_title(next_line)
                        and not self._is_numbered_section_title(next_line)
                        and len(next_line) <= 300
                    ):
                        caption = self._merge_caption(line, next_line)

                # 2) Если описание не нашлось в этом блоке —
                #    ищем центрированный блок, лежащий между этой
                #    «Таблица N» и верхом таблицы.
                if caption == line:
                    desc = self._find_caption_description_between(
                        ref_y_bottom=y1,
                        table_y_top=top,
                        page_text_blocks=page_text_blocks,
                    )
                    if desc:
                        caption = self._merge_caption(line, desc)

                return caption[:250]

        return None

    @staticmethod
    def _merge_caption(label: str, description: str) -> str:
        """Склеивает «Таблица N» и описание в один читаемый caption."""
        label = label.strip().rstrip(".")
        description = description.strip()
        if not description:
            return label
        return f"{label}. {description}"

    def _find_caption_description_between(
        self,
        ref_y_bottom: float,
        table_y_top: float,
        page_text_blocks: list[tuple[float, float, float, float, str]],
    ) -> str | None:
        """
        Ищет описание-таблицы — блок между нижней границей «Таблица N»
        и верхом самой таблицы. Не подбираем строки, похожие на section
        headings, табличные данные или другие caption.
        """

        best: tuple[float, str] | None = None
        for x0, y0, x1, y1, text in page_text_blocks:
            if y0 < ref_y_bottom:
                continue
            if y1 > table_y_top:
                continue

            first_line = ""
            for l in (text or "").splitlines():
                l = l.strip()
                if l:
                    first_line = l
                    break
            if not first_line:
                continue
            if self._is_table_title(first_line):
                continue
            if self._is_numbered_section_title(first_line):
                continue
            if self._is_table_data_line(first_line):
                continue

            # Берём верхний из подходящих (ближний к caption).
            if best is None or y0 < best[0]:
                best = (y0, first_line)

        return best[1] if best else None

    def _find_tables_raw(
        self,
        page,
        vertical_strategy: str,
        horizontal_strategy: str,
    ) -> list[dict]:
        """
        Запускает find_tables() с одной парой стратегий и возвращает
        список «сырых» таблиц с метрикой качества (score).

        score = число непустых ячеек * (1 + 0.5 * число колонок).
        Множитель по колонкам нужен, чтобы при одинаковом наборе ячеек
        вариант с 6 колонками побеждал вариант, склеенный в 2 колонки.
        """

        try:
            finder = page.find_tables(
                vertical_strategy=vertical_strategy,
                horizontal_strategy=horizontal_strategy,
            )
        except Exception:
            return []

        result: list[dict] = []

        for tab in list(getattr(finder, "tables", []) or []):
            try:
                raw_rows = tab.extract() or []
            except Exception:
                raw_rows = []

            if not raw_rows:
                continue

            rows = [
                [("" if c is None else str(c)) for c in row]
                for row in raw_rows
            ]

            non_empty = sum(
                1 for row in rows for cell in row if cell.strip()
            )
            # Скорее всего ложное срабатывание детектора.
            if non_empty < 4:
                continue

            ncols = max((len(row) for row in rows), default=0)

            # Штраф за «рваные слова»: когда vertical_strategy="text"
            # слишком агрессивно режет колонки и разрывает слова посередине
            # (напр. "огранич" | "енной ответственн" | "остью").
            fragmentation = self._fragmentation_ratio(rows)

            # Вклад числа колонок насыщается на 6 — больше колонок уже
            # не считается лучше (иначе метрика поощряет переразбиение).
            col_factor = 1.0 + 0.3 * min(ncols, 6)
            score = non_empty * col_factor * (1.0 - 0.7 * fragmentation)

            try:
                markdown = tab.to_markdown() or ""
            except Exception:
                markdown = ""

            if not markdown.strip() and not non_empty:
                continue

            result.append(
                {
                    "markdown": markdown,
                    "rows": rows,
                    "bbox": getattr(tab, "bbox", None),
                    "ncols": ncols,
                    "score": score,
                }
            )

        return result

    _WORD_CHAR_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")

    @classmethod
    def _fragmentation_ratio(cls, rows: list[list[str]]) -> float:
        """
        Оценивает, насколько таблица «разрезала слова».

        Возвращает долю пар соседних непустых ячеек в строке, где левая
        ячейка заканчивается буквой, а правая начинается со строчной буквы
        — это типичный признак того, что слово разорвано границей колонки
        (ложное переразбиение от vertical_strategy="text").

        0.0 — слова целые (нормальная таблица), ближе к 1.0 — таблица
        рассыпана на обрывки слов.
        """

        splits = 0
        pairs = 0

        for row in rows:
            cells = [c.strip() for c in row]
            for i in range(len(cells) - 1):
                left = cells[i]
                right = cells[i + 1]

                if not left or not right:
                    continue

                pairs += 1

                left_ends_letter = bool(cls._WORD_CHAR_RE.match(left[-1]))
                right_starts_lower = (
                    bool(cls._WORD_CHAR_RE.match(right[0])) and right[0].islower()
                )

                if left_ends_letter and right_starts_lower:
                    splits += 1

        if pairs == 0:
            return 0.0

        return splits / pairs

    @staticmethod
    def _page_text_excluding_tables(page, table_rects: list) -> str:
        """
        Возвращает текст страницы.

        Если на странице найдены таблицы — текст табличных зон вырезается,
        чтобы он не дублировался в обычных блоках. Если таблиц нет —
        ведём себя как раньше: простой page.get_text("text").
        """

        if not table_rects:
            return page.get_text("text") or ""

        parts: list[str] = []

        for block in page.get_text("blocks") or []:
            # block: (x0, y0, x1, y1, text, block_no, block_type)
            if len(block) < 5:
                continue

            x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
            text = block[4] or ""
            block_type = block[6] if len(block) > 6 else 0

            if block_type != 0:  # 0 = текстовый блок
                continue

            try:
                brect = fitz.Rect(x0, y0, x1, y1)
            except Exception:
                parts.append(text)
                continue

            brect_area = brect.get_area()
            inside_table = False

            for tr in table_rects:
                inter = brect & tr
                if inter.is_valid and brect_area > 0 and inter.get_area() > 0.5 * brect_area:
                    inside_table = True
                    break

            if inside_table:
                continue

            parts.append(text)

        return "\n".join(parts)

    @classmethod
    def _clean_section_candidate(cls, text: str) -> str | None:
        """
        Готовит heading-текст для использования как section_title.

        Что отсекает / правит:
          • псевдо-заголовки вида "Страница 16";
          • строки, которые ЦЕЛИКОМ выглядят как «Таблица N. …» — это
            caption таблицы, а не section;
          • «Рис. N» / «Рисунок N» по тем же причинам;
          • если внутри кандидата спрятан caption «… Таблица NN …» или
            «… Рис. NN …» (PyMuPDF склеил section heading и caption
            в один блок) — обрезаем кандидат до начала caption.
        """

        candidate = " ".join(text.split()).strip()

        if not candidate:
            return None

        if re.fullmatch(r"Страница\s*\d*", candidate, flags=re.IGNORECASE):
            return None

        # «Таблица N.» / «Таблица N Название» — это caption таблицы.
        if cls._is_table_title(candidate):
            return None

        # «Рис. N» / «Рисунок N» — caption рисунка.
        if re.match(r"^(рис(\.|унок)?)\s+\d+", candidate, flags=re.IGNORECASE):
            return None

        # Внутри кандидата лежит caption-хвост «… Таблица NN …» —
        # обрезаем до его начала. Не делаем для случая, когда «Таблица»
        # стоит в самом начале строки (это уже отрезано выше).
        m = re.search(
            r"\s+таблица\s+\d+(?:\.\d+)?\b",
            candidate,
            flags=re.IGNORECASE,
        )
        if m:
            candidate = candidate[: m.start()].rstrip(" .,;:-—–")

        # То же для встроенного «… Рис. NN …».
        m = re.search(
            r"\s+(рис(?:\.|унок)?)\s+\d+",
            candidate,
            flags=re.IGNORECASE,
        )
        if m:
            candidate = candidate[: m.start()].rstrip(" .,;:-—–")

        if len(candidate) < 4:
            return None

        return candidate

    def _metadata_with_parser(self, metadata: DocumentMetadata) -> DocumentMetadata:
        try:
            return replace(
                metadata,
                parser_name=self.parser_name,
                parser_version=self.parser_version,
            )
        except TypeError:
            return metadata

    # ============================================================
    # TEXT CLEANING
    # ============================================================

    @staticmethod
    def _clean_lines(text: str) -> list[str]:
        text = text.replace("\x00", " ")
        text = text.replace("\u00a0", " ")
        text = text.replace("\ufeff", " ")
        text = text.replace("\xad", "")

        lines: list[str] = []

        for line in text.splitlines():
            line = " ".join(line.split())
            line = line.strip()

            if not line:
                continue

            # Убираем одиночные номера страниц и мусорные короткие строки.
            if re.fullmatch(r"\d{1,3}", line):
                continue

            lines.append(line)

        return lines

    # ============================================================
    # PAGE BLOCKING
    # ============================================================

    def _build_page_blocks(
        self,
        lines: list[str],
        page_number: int,
        repeated_lines: set[str],
    ) -> list[dict[str, str]]:
        """
        Возвращает список блоков:
        [
            {"type": "heading", "text": "..."},
            {"type": "paragraph", "text": "..."},
        ]

        Важная идея:
        - заголовком становится только явный заголовок;
        - строки таблицы с числами НЕ становятся заголовками;
        - если страница похожа на таблицу, держим её более цельным блоком.
        """

        blocks: list[dict[str, str]] = []

        heading_lines, body_start = self._extract_page_heading(lines, repeated_lines)

        if heading_lines:
            blocks.append(
                {
                    "type": "heading",
                    "text": " ".join(heading_lines),
                    "strategy": "detected_heading",
                }
            )

        body_lines = lines[body_start:]

        if not body_lines:
            return blocks

        if self._looks_like_table_page(body_lines):
            table_like_chunks = self._chunk_lines(
                body_lines,
                max_lines=35,
                overlap_lines=4,
            )

            for chunk in table_like_chunks:
                blocks.append(
                    {
                        "type": "paragraph",
                        "text": "\n".join(chunk),
                        "strategy": "table_like_page_chunk",
                    }
                )

            return blocks

        paragraph_chunks = self._chunk_lines(
            body_lines,
            max_lines=22,
            overlap_lines=3,
        )

        for chunk in paragraph_chunks:
            blocks.append(
                {
                    "type": "paragraph",
                    "text": "\n".join(chunk),
                    "strategy": "page_text_chunk",
                }
            )

        return blocks

    def _extract_page_heading(
        self,
        lines: list[str],
        repeated_lines: set[str],
    ) -> tuple[list[str], int]:
        """
        Пытаемся выделить реальный заголовок страницы/таблицы.

        Типичный случай в табличных отчётах:
        Таблица 1.1
        Динамика основных показателей за 2020-2024 гг.
        Наименование показателя 2020 2021 ...

        Надо взять первые 1-2 строки, но НЕ захватывать строку с колонками таблицы.
        """

        if not lines:
            return [], 0

        # Пропускаем повторяющийся колонтитул.
        start = 0
        while start < min(len(lines), 3) and self._is_running_header(lines[start], repeated_lines):
            start += 1

        if start >= len(lines):
            return [], start

        first = lines[start]

        # Явный заголовок таблицы.
        if self._is_table_title(first):
            heading = [first]
            idx = start + 1

            # Добавляем 1-2 строки названия таблицы, но не строки с числами/годами.
            while idx < len(lines) and len(heading) < 3:
                candidate = lines[idx]

                if self._is_table_data_line(candidate):
                    break

                if self._is_column_header_line(candidate):
                    break

                if self._is_running_header(candidate, repeated_lines):
                    idx += 1
                    continue

                heading.append(candidate)
                idx += 1

            return heading, idx

        # Явный раздел: "1. Общие тенденции..."
        if self._is_numbered_section_title(first):
            heading = [first]
            idx = start + 1

            # Иногда заголовок перенесён на следующую строку.
            if idx < len(lines):
                candidate = lines[idx]
                if (
                    not self._is_table_data_line(candidate)
                    and not self._is_column_header_line(candidate)
                    and len(candidate) <= 140
                ):
                    heading.append(candidate)
                    idx += 1

            return heading, idx

        # Короткий текстовый заголовок без числового мусора.
        if self._is_clean_text_heading(first):
            return [first], start + 1

        # Если явного заголовка нет, ставим page heading.
        # Это лучше, чем использовать строку таблицы как section_title.
        return [f"Страница {self._extract_page_number_from_lines(lines) or ''}".strip()], start

    # ============================================================
    # HEURISTICS
    # ============================================================

    @staticmethod
    def _collect_repeated_lines(doc) -> set[str]:
        """
        Находит «обвязку» страниц — колонтитулы — универсально.

        Идея: колонтитул повторяется почти на каждой странице, а контент —
        нет. Собираем верхние и нижние строки каждой страницы, считаем, на
        скольких страницах встречается каждая строка, и берём те, что
        повторяются на значимой доле страниц. Без хардкода под документ.
        """

        page_count = getattr(doc, "page_count", 0)
        if page_count < 4:
            return set()

        counter: Counter[str] = Counter()

        for page_index in range(page_count):
            try:
                page = doc.load_page(page_index)
                text = page.get_text("text") or ""
            except Exception:
                continue

            lines = [" ".join(ln.split()) for ln in text.splitlines() if ln.strip()]
            if not lines:
                continue

            # Кандидаты в колонтитулы — верхние и нижние строки страницы.
            candidates = lines[:3] + lines[-3:]

            page_norms: set[str] = set()
            for line in candidates:
                norm = line.lower().strip()
                if not norm or norm.isdigit() or len(norm) > 120:
                    continue
                page_norms.add(norm)

            for norm in page_norms:
                counter[norm] += 1

        threshold = max(3, int(page_count * 0.4))
        return {line for line, count in counter.items() if count >= threshold}

    @staticmethod
    def _is_running_header(line: str, repeated_lines: set[str]) -> bool:
        """
        Колонтитул — строка, которая повторяется на многих страницах
        документа (см. _collect_repeated_lines). Никаких строк, привязанных
        к конкретному отчёту, — логика универсальна.
        """

        norm = " ".join(line.split()).lower().strip()
        return bool(norm) and norm in repeated_lines

    @staticmethod
    def _is_table_title(line: str) -> bool:
        return bool(re.match(r"^таблица\s+\d+(\.\d+)?\b", line.strip(), flags=re.IGNORECASE))

    @staticmethod
    def _is_numbered_section_title(line: str) -> bool:
        line = line.strip()

        # 1. Общие тенденции...
        if re.match(r"^\d+(\.\d+)*\.\s+\D", line):
            return len(line) <= 180

        return False

    def _is_clean_text_heading(self, line: str) -> bool:
        line = line.strip()

        if len(line) < 5 or len(line) > 140:
            return False

        if self._is_table_data_line(line):
            return False

        if self._is_column_header_line(line):
            return False

        numeric_tokens = re.findall(r"\d+", line)
        if len(numeric_tokens) >= 3:
            return False

        # Частые заголовки в отчетах.
        upper_like = sum(1 for ch in line if ch.isupper())
        letters = sum(1 for ch in line if ch.isalpha())

        if letters > 0 and upper_like / letters > 0.55:
            return True

        # Универсальные структурные заголовки аналитических отчётов
        # (не привязаны к теме документа).
        keywords = [
            "предисловие",
            "введение",
            "содержание",
            "оглавление",
            "общие тенденции",
            "основные причины",
            "распределение",
            "динамика",
            "выводы",
            "заключение",
        ]

        norm = line.lower().replace("ё", "е")

        return any(keyword in norm for keyword in keywords)

    @staticmethod
    def _is_column_header_line(line: str) -> bool:
        """
        Универсально определяет строку-«шапку таблицы».

        Признак: в строке стоит несколько годов подряд (2020 2021 2022 …) —
        типичная шапка временного ряда. Без привязки к названиям колонок
        конкретного отчёта.
        """

        year_count = len(re.findall(r"\b(?:19|20)\d{2}\b", line))
        return year_count >= 3

    @staticmethod
    def _is_table_data_line(line: str) -> bool:
        numbers = re.findall(r"-?\d+(?:[.,]\d+)?", line)

        if len(numbers) >= 4:
            return True

        # Строка с годами и показателями.
        if len(re.findall(r"\b20\d{2}\b", line)) >= 2:
            return True

        return False

    def _looks_like_table_page(self, lines: list[str]) -> bool:
        if not lines:
            return False

        sample = lines[:80]

        table_data_lines = sum(1 for line in sample if self._is_table_data_line(line))
        column_lines = sum(1 for line in sample if self._is_column_header_line(line))
        table_title_lines = sum(1 for line in sample if self._is_table_title(line))

        ratio = table_data_lines / max(1, len(sample))

        return (
            table_title_lines > 0
            or column_lines > 0
            or ratio >= 0.25
        )

    @staticmethod
    def _extract_page_number_from_lines(lines: list[str]) -> str | None:
        for line in lines[:3]:
            m = re.search(r"\b(\d{1,3})\b", line)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _chunk_lines(
        lines: list[str],
        max_lines: int,
        overlap_lines: int = 0,
    ) -> list[list[str]]:
        if not lines:
            return []

        chunks: list[list[str]] = []
        start = 0

        while start < len(lines):
            end = min(len(lines), start + max_lines)
            chunk = lines[start:end]

            if chunk:
                chunks.append(chunk)

            if end >= len(lines):
                break

            start = max(0, end - overlap_lines)

        return chunks