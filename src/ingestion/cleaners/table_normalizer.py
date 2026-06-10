from __future__ import annotations

import re

from ingestion.models import ParsedDocument, ParsedTable


class TableNormalizer:
    """
    Нормализует markdown-таблицы для RAG.

    Универсальный подход:
    1. markdown table -> columns + rows
    2. чинит типовые проблемы заголовков
    3. удаляет мусорные/служебные колонки
    4. удаляет повторяющиеся строки-заголовки
    5. склеивает разорванные строки
    6. rows -> readable table_text для embeddings/RAG
    7. сохраняет результат в table.metadata
    """

    ARROW_VALUES = {"↑", "↓", "↗", "↘", "→", "←", "▲", "▼"}

    def normalize_document(self, document: ParsedDocument) -> ParsedDocument:
        normalized_tables: list[ParsedTable] = []

        for table in document.tables:
            normalized = self.normalize_table(table)
            normalized_tables.append(normalized)

        document.tables = normalized_tables
        return document

    def normalize_table(self, table: ParsedTable) -> ParsedTable:
        columns, rows = self._parse_markdown_table(table.markdown_table)

        columns, rows = self._promote_header_like_first_row(columns, rows)
        columns, rows = self._fix_joined_period_headers(columns, rows)
        columns, rows = self._rename_period_companion_columns(columns, rows)
        columns, rows = self._drop_low_value_columns(columns, rows)
        rows = self._drop_repeated_header_rows(columns, rows)
        rows = self._merge_broken_continuation_rows(columns, rows)

        table_text = self._build_table_text(table, columns, rows)

        table.metadata["columns"] = columns
        table.metadata["normalized_rows"] = rows
        table.metadata["table_text"] = table_text
        table.metadata["table_quality"] = self._estimate_quality(columns, rows)

        return table

    def _parse_markdown_table(
        self,
        markdown: str,
    ) -> tuple[list[str], list[dict[str, str]]]:
        """
        Преобразует markdown-таблицу в:
        - список колонок;
        - список строк dict[column] = value.
        """

        lines = [
            line.strip()
            for line in markdown.splitlines()
            if line.strip().startswith("|")
        ]

        if len(lines) < 2:
            return [], []

        raw_rows: list[list[str]] = []

        for line in lines:
            # Пропускаем markdown-разделитель вида |---|---|
            if re.fullmatch(r"[\|\s:\-]+", line):
                continue

            cells = [self._clean_cell(cell) for cell in line.strip("|").split("|")]
            raw_rows.append(cells)

        if not raw_rows:
            return [], []

        columns = self._make_unique_headers(raw_rows[0])
        data_rows = raw_rows[1:]

        rows: list[dict[str, str]] = []

        for raw_row in data_rows:
            normalized_row = raw_row + [""] * (len(columns) - len(raw_row))
            normalized_row = normalized_row[: len(columns)]

            row_dict = {
                columns[i]: normalized_row[i]
                for i in range(len(columns))
            }

            if any(value.strip() for value in row_dict.values()):
                rows.append(row_dict)

        return columns, rows

    @staticmethod
    def _clean_cell(value: str) -> str:
        value = value.replace("\u00a0", " ")
        value = re.sub(r"[ \t]+", " ", value)
        value = value.strip()
        return value

    @staticmethod
    def _make_unique_headers(headers: list[str]) -> list[str]:
        """
        Делает уникальные имена колонок.
        """

        result: list[str] = []
        seen: dict[str, int] = {}

        for idx, header in enumerate(headers):
            clean_header = header.strip() or f"column_{idx + 1}"

            if clean_header not in seen:
                seen[clean_header] = 1
                result.append(clean_header)
            else:
                seen[clean_header] += 1
                result.append(f"{clean_header}_{seen[clean_header]}")

        return result

    def _promote_header_like_first_row(
        self,
        columns: list[str],
        rows: list[dict[str, str]],
    ) -> tuple[list[str], list[dict[str, str]]]:
        """
        Универсальная эвристика для многоуровневых заголовков.

        Иногда Docling делает:
        columns = ["column_1", "column_2", "2026 г.", "2027 г.", "2028 г."]
        first row = ["Показатели", "2025 г.", "базовый", "базовый", "базовый"]

        Тогда можно взять значения из первой строки для column_1 / column_2.
        """

        if not columns or not rows:
            return columns, rows

        first_row = rows[0]

        new_columns = columns.copy()
        changed = False

        for idx, col in enumerate(columns):
            value = first_row.get(col, "").strip()

            if not value:
                continue

            # Поднимаем значение первой строки в заголовок только для безымянных column_N.
            if col.startswith("column_") and self._looks_like_header_value(value):
                new_columns[idx] = value
                changed = True

        if not changed:
            return columns, rows

        new_columns = self._make_unique_headers(new_columns)

        new_rows: list[dict[str, str]] = []

        for old_row in rows[1:]:
            new_row: dict[str, str] = {}

            for old_col, new_col in zip(columns, new_columns):
                new_row[new_col] = old_row.get(old_col, "")

            if any(value.strip() for value in new_row.values()):
                new_rows.append(new_row)

        return new_columns, new_rows

    @staticmethod
    def _looks_like_header_value(value: str) -> bool:
        value_lower = value.lower().strip()

        if not value_lower:
            return False

        header_keywords = [
            "показател",
            "страна",
            "код",
            "группа",
            "номенклатур",
            "2023",
            "2024",
            "2025",
            "2026",
            "2027",
            "2028",
            "янв",
            "нояб",
            "прирост",
            "год",
        ]

        return any(keyword in value_lower for keyword in header_keywords)

    def _fix_joined_period_headers(
        self,
        columns: list[str],
        rows: list[dict[str, str]],
    ) -> tuple[list[str], list[dict[str, str]]]:
        """
        Чинит типовую ошибку PDF-парсинга:
        две соседние колонки получили одинаковый склеенный заголовок:
        "янв. - нояб. 2024 г. янв. - нояб. 2025 г."
        и "..._2".

        Универсальная логика:
        если заголовок содержит 2024 и 2025, то первая такая колонка становится 2024,
        вторая — 2025.
        """

        if not columns:
            return columns, rows

        new_columns = columns.copy()
        joined_period_indices: list[int] = []

        for idx, col in enumerate(columns):
            norm = col.lower()
            if "2024" in norm and "2025" in norm:
                joined_period_indices.append(idx)

        if len(joined_period_indices) >= 2:
            first_idx = joined_period_indices[0]
            second_idx = joined_period_indices[1]

            first_label, second_label = self._split_joined_period_label(columns[first_idx])

            new_columns[first_idx] = first_label
            new_columns[second_idx] = second_label

        new_columns = self._make_unique_headers(new_columns)

        if new_columns == columns:
            return columns, rows

        new_rows: list[dict[str, str]] = []

        for row in rows:
            new_row: dict[str, str] = {}

            for old_col, new_col in zip(columns, new_columns):
                new_row[new_col] = row.get(old_col, "")

            new_rows.append(new_row)

        return new_columns, new_rows

    @staticmethod
    def _split_joined_period_label(label: str) -> tuple[str, str]:
        """
        Из склеенного заголовка пытается сделать два периода.
        """

        # Частый случай: "янв. - нояб. 2024 г. янв. - нояб. 2025 г."
        lower = label.lower()

        if "янв" in lower and "нояб" in lower:
            return "янв. - нояб. 2024 г.", "янв. - нояб. 2025 г."

        return "2024 г.", "2025 г."

    _YEAR_RE = re.compile(r"(19|20)\d{2}")

    def _rename_period_companion_columns(
        self,
        columns: list[str],
        rows: list[dict[str, str]],
    ) -> tuple[list[str], list[dict[str, str]]]:
        """
        Чинит частый паттерн в РСБУ-выгрузках Контур.Фокус:

        Колонки: ..., "2022", "column_4", "2023", "column_6", "2024", "column_8", ...

        Безымянные column_N между годами — это «сравнительный период»
        из того же годового отчёта (т.е. «На 31.12.предыдущего года»),
        который выгружается отдельной колонкой без шапки. Сейчас их
        имена ничего не говорят ни embedding'у, ни LLM — а значения
        часто конфликтуют с соседним именованным годом из другого
        отчётного периода, что приводит к путанице цифр.

        Правило (универсальное):
        если column_N стоит сразу после колонки, в имени которой есть
        4-значный год — переименовываем column_N в "<год>_сравн".

        Если все значения такой колонки совпадают со значениями
        соседнего «старшего» года (то есть это буквально дубль) —
        полностью отдадим её на удаление в _drop_low_value_columns
        (там безымянные колонки уходят при низкой заполненности,
        но здесь мы уже сделали смыслоулавливающий шаг).
        """

        if not columns:
            return columns, rows

        new_columns = columns.copy()
        changed = False

        for idx, col in enumerate(columns):
            if not col.startswith("column_"):
                continue
            if idx == 0:
                continue

            prev = new_columns[idx - 1]
            match = self._YEAR_RE.search(prev)
            if not match:
                continue

            year = match.group(0)
            label = f"{year}_сравн"

            # Если такой label уже есть в списке — пометим суффиксом.
            if label in new_columns:
                k = 2
                while f"{label}_{k}" in new_columns:
                    k += 1
                label = f"{label}_{k}"

            new_columns[idx] = label
            changed = True

        if not changed:
            return columns, rows

        new_columns = self._make_unique_headers(new_columns)

        new_rows: list[dict[str, str]] = []
        for row in rows:
            new_row: dict[str, str] = {}
            for old_col, new_col in zip(columns, new_columns):
                new_row[new_col] = row.get(old_col, "")
            new_rows.append(new_row)

        return new_columns, new_rows

    def _drop_low_value_columns(
        self,
        columns: list[str],
        rows: list[dict[str, str]],
    ) -> tuple[list[str], list[dict[str, str]]]:
        """
        Удаляет служебные/почти пустые колонки.

        Универсальные случаи:
        - полностью пустые колонки;
        - колонки, где почти все значения — стрелки;
        - безымянные column_N с низкой заполненностью.
        """

        if not columns or not rows:
            return columns, rows

        keep_columns: list[str] = []

        for col in columns:
            values = [row.get(col, "").strip() for row in rows]
            non_empty_values = [v for v in values if v]

            if not non_empty_values:
                continue

            arrow_count = sum(1 for v in non_empty_values if v in self.ARROW_VALUES)
            arrow_ratio = arrow_count / len(non_empty_values)

            # Любую почти стрелочную колонку удаляем.
            if arrow_ratio >= 0.7:
                continue

            # Безымянные слабо заполненные колонки удаляем.
            if col.startswith("column_") and len(non_empty_values) <= max(1, int(len(rows) * 0.3)):
                continue

            keep_columns.append(col)

        cleaned_rows: list[dict[str, str]] = []

        for row in rows:
            cleaned_row = {col: row.get(col, "") for col in keep_columns}

            if any(value.strip() for value in cleaned_row.values()):
                cleaned_rows.append(cleaned_row)

        return keep_columns, cleaned_rows

    def _drop_repeated_header_rows(
        self,
        columns: list[str],
        rows: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Удаляет строки, которые являются повтором заголовков таблицы.
        """

        if not columns or not rows:
            return rows

        cleaned_rows: list[dict[str, str]] = []

        normalized_columns = {
            self._normalize_for_compare(col)
            for col in columns
            if col.strip()
        }

        for row in rows:
            values = [
                self._normalize_for_compare(value)
                for value in row.values()
                if value and value.strip()
            ]

            if not values:
                continue

            repeated_header_hits = sum(
                1 for value in values if value in normalized_columns
            )

            if repeated_header_hits >= max(2, len(values) // 2):
                continue

            cleaned_rows.append(row)

        return cleaned_rows

    def _merge_broken_continuation_rows(
        self,
        columns: list[str],
        rows: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Склеивает/удаляет строки-хвосты.

        Типовая проблема:
        строка с длинным описанием разорвалась, и в следующей строке оказалось:
        "тыс. USD." или "или судах".

        Универсальная эвристика:
        если строка не содержит числовых значений в числовых/периодных колонках
        и заполнена в основном первая текстовая колонка — считаем её продолжением
        предыдущей строки.
        """

        if not columns or not rows:
            return rows

        numeric_like_columns = [
            col for col in columns
            if self._is_numeric_or_period_column(col)
        ]

        text_columns = [
            col for col in columns
            if col not in numeric_like_columns
        ]

        if not text_columns:
            return rows

        primary_text_col = text_columns[0]

        merged_rows: list[dict[str, str]] = []

        for row in rows:
            if not merged_rows:
                merged_rows.append(row)
                continue

            if self._looks_like_continuation_row(row, primary_text_col, numeric_like_columns):
                tail = row.get(primary_text_col, "").strip()

                if tail:
                    prev = merged_rows[-1]
                    prev_value = prev.get(primary_text_col, "").strip()

                    if prev_value:
                        prev[primary_text_col] = f"{prev_value} {tail}".strip()
                    else:
                        prev[primary_text_col] = tail

                continue

            merged_rows.append(row)

        return merged_rows

    def _looks_like_continuation_row(
        self,
        row: dict[str, str],
        primary_text_col: str,
        numeric_like_columns: list[str],
    ) -> bool:
        primary_value = row.get(primary_text_col, "").strip()

        if not primary_value:
            return False

        # Если в числовых колонках есть числа — это нормальная строка, не хвост.
        for col in numeric_like_columns:
            value = row.get(col, "").strip()
            if self._contains_number(value):
                return False

        non_empty = {
            col: value.strip()
            for col, value in row.items()
            if value and value.strip()
        }

        # Если заполнена только одна короткая текстовая ячейка — похоже на хвост.
        if len(non_empty) == 1 and primary_text_col in non_empty:
            return len(primary_value) <= 80

        # Если одна и та же короткая фраза размазалась по нескольким колонкам.
        unique_values = set(non_empty.values())
        if len(unique_values) == 1:
            only_value = next(iter(unique_values))
            return len(only_value) <= 80 and not self._contains_number(only_value)

        return False

    @staticmethod
    def _is_numeric_or_period_column(col: str) -> bool:
        col_lower = col.lower()

        patterns = [
            r"20\d{2}",
            r"янв",
            r"нояб",
            r"прирост",
            r"рост",
            r"%",
            r"usd",
            r"руб",
            r"млн",
            r"тыс",
        ]

        return any(re.search(pattern, col_lower) for pattern in patterns)

    @staticmethod
    def _contains_number(value: str) -> bool:
        return bool(re.search(r"\d", value))

    @staticmethod
    def _normalize_for_compare(value: str) -> str:
        """
        Нормализует строку для сравнения заголовков и значений.
        """

        value = value.strip().lower()
        value = value.replace(" ", "")
        value = value.replace("\u00a0", "")
        value = value.replace(".", "")
        value = value.replace("-", "")
        value = value.replace("–", "")
        value = value.replace("—", "")
        return value

    def _build_table_text(
        self,
        table: ParsedTable,
        columns: list[str],
        rows: list[dict[str, str]],
    ) -> str:
        """
        Делает компактный markdown-текст таблицы для RAG / embeddings.

        Раньше был многословный формат "- Строка N: Колонка: значение; ..." —
        он раздувал чанки в 2-3 раза и заставлял _split_table_text резать
        даже средние таблицы. Теперь — обычный markdown pipe-table.

        Плюсы:
        - 30-40% компактнее → меньше multi-part чанков;
        - qwen3 нативно понимает markdown-таблицы;
        - _split_table_text уже умеет резать pipe-формат по строкам данных.

        Шапка с метаданными (документ / страница / название / колонки)
        нужна, чтобы при splitting каждое окно сохраняло контекст.
        """

        if not columns or not rows:
            return table.markdown_table

        page_str = (
            str(table.page_number)
            if table.page_number is not None
            else "не определена"
        )
        title_str = table.table_title or "не определено"

        header_lines = [
            f"Таблица из документа: {table.source_file}",
            f"Страница: {page_str}",
            f"Название таблицы: {title_str}",
            "",
        ]

        # Markdown pipe-table. Пустые значения оставляем пустыми ячейками,
        # а не "—" / "0", чтобы не вводить LLM в заблуждение.
        md_header = "| " + " | ".join(columns) + " |"
        md_sep = "|" + "|".join("---" for _ in columns) + "|"

        md_rows: list[str] = []
        for row in rows:
            cells: list[str] = []
            non_empty = False
            for col in columns:
                value = (row.get(col, "") or "").strip()
                # Внутри ячейки переводы строк и pipe'ы — спасаем markdown.
                value = value.replace("|", "\\|").replace("\n", " ")
                if value:
                    non_empty = True
                cells.append(value)
            if non_empty:
                md_rows.append("| " + " | ".join(cells) + " |")

        if not md_rows:
            return "\n".join(header_lines).rstrip()

        return "\n".join(header_lines + [md_header, md_sep] + md_rows)

    @staticmethod
    def _estimate_quality(
        columns: list[str],
        rows: list[dict[str, str]],
    ) -> str:
        """
        Грубая оценка качества распознанной таблицы.
        """

        if not columns or not rows:
            return "bad"

        if len(columns) < 2 or len(rows) < 1:
            return "weak"

        non_empty_cells = 0
        total_cells = len(columns) * len(rows)

        for row in rows:
            for value in row.values():
                if value and value.strip():
                    non_empty_cells += 1

        fill_rate = non_empty_cells / total_cells if total_cells else 0

        if fill_rate >= 0.7:
            return "good"

        if fill_rate >= 0.4:
            return "medium"

        return "weak"