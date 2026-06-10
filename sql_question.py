"""
Text-to-SQL: менеджер задаёт вопрос на русском, LLM (Ollama) превращает
его в SELECT по схеме finance.*, Postgres исполняет, результат печатается
таблицей. По флагу --explain LLM ещё кратко резюмирует данные.

Это параллельный RAG-у путь: для качественных вопросов про документы —
answer_question.py (RAG), для точных финансовых выгрузок — sql_question.py.

Пример:
    python sql_question.py "Покажи выручку АО Сибкабель за 2024"
    python sql_question.py "Сравни чистую прибыль НЛМК и ЕВРАЗ за 2023 и 2024"
    python sql_question.py "Топ-5 по балансу на конец 2024" --explain
    python sql_question.py "Какие компании показали убыток в 2024?" --dry-run

Безопасность:
- Принимается только SELECT/WITH; INSERT/UPDATE/DELETE/DROP/... запрещены
  валидатором ещё ДО исполнения.
- Транзакция помечается READ ONLY на уровне Postgres — второй контур защиты.
- Жёсткий statement_timeout (по умолчанию 10 сек).
- Авто-LIMIT, если LLM забыл указать (по умолчанию 100 строк).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL = "qwen3:14b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_LIMIT = 100
DEFAULT_TIMEOUT_MS = 10_000


# ============================================================
# FEW-SHOT
# ============================================================

FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    (
        "Какая выручка у АО Сибкабель за 2024 год?",
        "SELECT c.short_name, i.reporting_year, "
        "i.c_2110_vyruchka AS vyruchka_tys_rub\n"
        "FROM finance.income_statement i\n"
        "JOIN finance.company c ON c.id = i.company_id\n"
        "WHERE c.short_name ILIKE '%сибкабель%' AND i.reporting_year = 2024",
    ),
    (
        "Сравни чистую прибыль ПАО НЛМК за 2023 и 2024",
        "SELECT c.short_name, i.reporting_year, "
        "i.c_2400_chistaya_pribyl AS chistaya_pribyl_tys_rub\n"
        "FROM finance.income_statement i\n"
        "JOIN finance.company c ON c.id = i.company_id\n"
        "WHERE c.short_name ILIKE '%нлмк%' AND i.reporting_year IN (2023, 2024)\n"
        "ORDER BY i.reporting_year",
    ),
    (
        "Топ-5 компаний по балансу на конец 2024 года",
        "SELECT c.short_name, b.c_1600_balans AS balans_tys_rub\n"
        "FROM finance.balance_sheet b\n"
        "JOIN finance.company c ON c.id = b.company_id\n"
        "WHERE b.reporting_year = 2024 AND b.period_date = '2024-12-31'\n"
        "ORDER BY b.c_1600_balans DESC NULLS LAST\n"
        "LIMIT 5",
    ),
    (
        "Какие компании показали чистый убыток в 2024?",
        "SELECT c.short_name, "
        "i.c_2400_chistaya_pribyl AS chistaya_pribyl_tys_rub\n"
        "FROM finance.income_statement i\n"
        "JOIN finance.company c ON c.id = i.company_id\n"
        "WHERE i.reporting_year = 2024 AND i.c_2400_chistaya_pribyl < 0\n"
        "ORDER BY i.c_2400_chistaya_pribyl",
    ),
    (
        "Сравни Asset Turnover АО ЭКЗ на конец 2022 и 2024",
        "SELECT c.short_name, i.reporting_year,\n"
        "       i.c_2110_vyruchka AS vyruchka_tys_rub,\n"
        "       b.c_1600_balans  AS aktivy_tys_rub,\n"
        "       ROUND((i.c_2110_vyruchka::numeric / NULLIF(b.c_1600_balans, 0)), 3) AS asset_turnover\n"
        "FROM finance.income_statement i\n"
        "JOIN finance.balance_sheet  b ON b.company_id = i.company_id\n"
        "                              AND b.reporting_year = i.reporting_year\n"
        "                              AND b.period_date = (i.reporting_year || '-12-31')::date\n"
        "JOIN finance.company c ON c.id = i.company_id\n"
        "WHERE c.short_name ILIKE '%экз%' AND i.reporting_year IN (2022, 2024)\n"
        "ORDER BY i.reporting_year",
    ),
    (
        "Топ-3 кабельщиков по выручке за 2024",
        "SELECT c.short_name, c.segment, i.c_2110_vyruchka AS vyruchka_tys_rub\n"
        "FROM finance.income_statement i\n"
        "JOIN finance.company c ON c.id = i.company_id\n"
        "WHERE c.segment = 'Кабельное производство' AND i.reporting_year = 2024\n"
        "ORDER BY i.c_2110_vyruchka DESC NULLS LAST\n"
        "LIMIT 3",
    ),
    (
        "Сравни выручку, прибыль от продаж и долю прибыли в выручке по сегментам за 2024",
        "SELECT c.segment,\n"
        "       COUNT(*) AS companies,\n"
        "       SUM(i.c_2110_vyruchka)        AS vyruchka_tys_rub,\n"
        "       SUM(i.c_2200_pribyl_prodazh)  AS pribyl_prodazh_tys_rub,\n"
        "       ROUND(SUM(i.c_2200_pribyl_prodazh)::numeric\n"
        "             / NULLIF(SUM(i.c_2110_vyruchka), 0) * 100, 2) AS op_margin_pct\n"
        "FROM finance.income_statement i\n"
        "JOIN finance.company c ON c.id = i.company_id\n"
        "WHERE i.reporting_year = 2024 AND c.segment IS NOT NULL\n"
        "GROUP BY c.segment\n"
        "ORDER BY vyruchka_tys_rub DESC NULLS LAST",
    ),
]


def format_examples() -> str:
    parts = []
    for question, sql in FEW_SHOT_EXAMPLES:
        parts.append(f"Вопрос: {question}\nSQL:\n{sql}\n")
    return "\n".join(parts)


# ============================================================
# PROMPT
# ============================================================

PROMPT_TEMPLATE = """/no_think

Ты помощник, который превращает вопросы аналитика на русском в SQL-запросы
к PostgreSQL. Работаешь только со схемой finance.

⚠️ САМОЕ ВАЖНОЕ — JOIN ТАБЛИЦ ОТЧЁТНОСТИ ⚠️
finance.balance_sheet содержит ДВЕ записи на год: 'YYYY-01-01' (начало)
и 'YYYY-12-31' (конец). finance.income_statement содержит ОДНУ запись.

Если делаешь JOIN income_statement + balance_sheet БЕЗ фильтра по
period_date — каждый год удвоится. Это ОШИБКА.

ВСЕГДА когда соединяешь эти две таблицы, в условии JOIN добавляй:
    AND b.period_date = (b.reporting_year || '-12-31')::date

Или, если вопрос ТОЛЬКО про доходы/прибыль (выручка, прибыль от продаж,
чистая прибыль, любой показатель из c_21XX/c_22XX/c_23XX/c_24XX) —
НЕ ДЕЛАЙ JOIN с balance_sheet вообще. Возьми данные только из
finance.income_statement.

ПРАВИЛЬНЫЙ ПРИМЕР JOIN (когда нужны и доходы и активы):
    FROM finance.income_statement i
    JOIN finance.balance_sheet b ON b.company_id = i.company_id
                                 AND b.reporting_year = i.reporting_year
                                 AND b.period_date = (b.reporting_year || '-12-31')::date
    JOIN finance.company c ON c.id = i.company_id

НЕПРАВИЛЬНО (даст дубли):
    JOIN finance.balance_sheet b ON b.company_id = i.company_id
                                 AND b.reporting_year = i.reporting_year
    -- ↑ нет period_date → 2 строки на год

Схема БД:
{schema}

{companies}
{glossary}
Правила:
- Только SELECT (или WITH ... SELECT). Никаких INSERT / UPDATE / DELETE /
  DROP / ALTER / TRUNCATE / CREATE — это запрещено.
- Один statement, без точки с запятой в конце.
- Имена компаний в finance.company.short_name содержат ПОДЧЁРКИВАНИЯ
  вместо пробелов: «АО_СИБКАБЕЛЬ», «ПАО_НЛМК», «ООО_АЛПИНА» и т.п.
  Поэтому ищи через ILIKE с %, БЕЗ префикса юр. формы и БЕЗ пробелов:
    ILIKE '%сибкабель%'   (а НЕ '%АО СИБКАБЕЛЬ%')
    ILIKE '%нлмк%'        (а НЕ '%ПАО НЛМК%')
    ILIKE '%экз%'         (а НЕ '%АО ЭКЗ%')
  Если в вопросе НЕСКОЛЬКО компаний — соединяй через OR:
    WHERE (c.short_name ILIKE '%сибкабель%' OR c.short_name ILIKE '%нлмк%')
- Все суммы в БД хранятся в тыс. руб. (так выгружает Контур.Фокус).
  В самом SELECT не делай арифметику пересчёта в млн/млрд — выводи как есть.
  Пересчёт сделает SUMMARY-промпт при показе пользователю.
- Когда вопрос про коэффициент из «Справочника финансовых показателей» —
  используй ИМЕННО ту формулу, что в справочнике. Не выбирай похожую
  колонку «на глаз» (например, Operating Margin = c_2200 / c_2110, а НЕ
  c_2300 / c_2110 — последнее это Pretax Margin).
- КРИТИЧНО про JOIN income_statement + balance_sheet:
  balance_sheet хранит ДВЕ записи на год (period_date 'YYYY-01-01' = начало,
  'YYYY-12-31' = конец). Если делаешь JOIN без фильтра по period_date —
  получишь ДУБЛИ строк (каждый год повторится дважды).
  ВСЕГДА указывай в JOIN: AND b.period_date = (b.reporting_year || '-12-31')::date
  (или иной конкретный snapshot по контексту вопроса).
  Если вопрос ТОЛЬКО про ОФР (выручка, прибыль) — НЕ делай JOIN с balance_sheet
  вовсе, оставайся на одной finance.income_statement.
- Если в начале вопроса есть блок «АКТИВНЫЙ КОНТЕКСТ» с конкретной
  компанией или периодом — используй ИХ как фильтры в WHERE по умолчанию.
  Не возвращай данные по всем компаниям, если пользователь явно не сказал
  «по всем». Follow-up («а Quick Ratio упал из-за чего?») продолжает
  фокус на той же компании, что и предыдущий вопрос.
- В таблице finance.company есть колонка `segment` (отрасль).
  ⚠️ ВСЕГДА фильтруй по сегменту через ТОЧНОЕ РАВЕНСТВО:
        c.segment = '<точное_значение>'
  НЕ используй ILIKE для сегмента — это часто приводит к подстановке
  обрывка имени компании («%сибкабель%» вместо «Кабельное производство»).
  Значения колонки sgment (исчерпывающий список, копируй БУКВА В БУКВУ,
  включая регистр, пробелы, скобки и слэши):
    • 'Кабельное производство'
    • 'Компаунды/Полимеры'
    • 'Кормовые добавки'
    • 'Молочная промышленность (раскислитель)'
    • 'АКП'
    • 'Удобрения/Агрохимикаты'
    • 'Металлургия/Огнеупорные материалы/Флюс'
    • 'РТИ'
    • 'Дистрибьютер'
    • 'Переработчики (B2B)'
  Жаргонный мэппинг русского запроса → точное значение из БД:
    • «кабельщики» → 'Кабельное производство'
    • «дистры», «дистрибы», «дистрибьютеры» → 'Дистрибьютер'
    • «молочка», «молочники» → 'Молочная промышленность (раскислитель)'
    • «огнеупоры», «металлурги», «металлургия» → 'Металлургия/Огнеупорные материалы/Флюс'
    • «полимерщики», «компаундщики», «компаунды» → 'Компаунды/Полимеры'
    • «корма», «кормовики» → 'Кормовые добавки'
    • «АКП» → 'АКП'
    • «удобренцы», «агрохимия» → 'Удобрения/Агрохимикаты'
    • «РТИ», «резинщики» → 'РТИ'
    • «переработчики», «B2B-переработчики» → 'Переработчики (B2B)'
  Сводки/сравнения ВСЕХ сегментов делай через GROUP BY c.segment
  (тогда WHERE по сегменту не нужен).
- Алиасы колонок давай человеческие, например vyruchka_tys_rub, balans_tys_rub,
  operating_margin_pct.
- Для процентов ROUND до 2 знаков: ROUND(...::numeric, 2).
- 🚫 НИКОГДА не вкладывай агрегатные функции друг в друга. Postgres
  кинет ошибку «aggregate function calls cannot be nested». Запрещены
  конструкции вроде SUM(x / NULLIF(SUM(y), 0)) или AVG(SUM(x)).
  ПРАВИЛЬНЫЙ паттерн для отношения сумм внутри GROUP BY (например,
  «общая прибыль / общая выручка по сегменту»):
      ROUND(SUM(i.c_2200_pribyl_prodazh)::numeric
            / NULLIF(SUM(i.c_2110_vyruchka), 0) * 100, 2)
            AS operating_margin_pct
  Агрегаты SUM/AVG/COUNT берут «голые» колонки в аргументе, а
  деление/умножение делаешь СНАРУЖИ агрегатов. Если же нужен «средний
  показатель» (а не отношение сумм) — используй AVG(x / NULLIF(y, 0))
  без вложенного SUM/AVG в y.
- Если LIMIT уместен, ставь не больше {limit}.
- В балансе период различай period_date: 'YYYY-01-01' = Начало года,
  'YYYY-12-31' = Конец года. Если запрос про «на конец 2024» — period_date = '2024-12-31'.
- Возвращай ТОЛЬКО SQL, без комментариев и без оборачивания в ```.

Примеры:
{examples}

Вопрос: {question}
SQL:"""


SUMMARY_PROMPT_TEMPLATE = """/no_think

Ты аналитик. По вопросу пользователя и результату SQL дай ответ на русском.

🚫 РАБОТАЙ ТОЛЬКО С ТЕМ, ЧТО В ДАННЫХ 🚫
- Используй ТОЛЬКО колонки из «Колонки результата». Нет колонки — нет темы:
  если в данных лишь vyruchka_tys_rub, пиши только про выручку, не упоминай
  прибыль, активы, обязательства и пр.
- НЕ выдумывай числа. Каждое число берётся из строк дословно. Нет данных за
  год — не пиши его.
- Одна строка = одна сущность (компания + период). Несколько РАЗНЫХ компаний
  с ОДНИМ reporting_year — это рейтинг, НЕ динамика: не строй цепочку X→Y→Z.
  Цепочка-динамика уместна ТОЛЬКО когда один short_name и разные reporting_year.

⚠️ ОТРИЦАТЕЛЬНАЯ «ПРИБЫЛЬ» = УБЫТОК ⚠️
Колонки c_2200_pribyl_prodazh, c_2300_pribyl_do_nalog, c_2400_chistaya_pribyl
называются «Прибыль», но при ОТРИЦАТЕЛЬНОМ значении это УБЫТОК.
- Есть минус → пиши слово «убыток» и значение БЕЗ минуса (слово уже несёт
  отрицательный смысл):
    c_2400 = −129093  →  «**Чистый убыток: 129,1 млн руб.**»
    (а НЕ «Чистая прибыль: −0,13 млрд руб.» и не «убыток −129,1»).
- Минуса нет → это ПРИБЫЛЬ, убытком не называй. Перед словом «убыток» всегда
  проверь, реально ли в данных стоит минус.
- В динамике при смене знака покажи переход явно:
    «2,12 → 1,66 → 1,22 млрд → **в 2025 ушла в убыток 133,9 млн руб.**».
- Если спрашивали про убыток/проблемы, а данные положительные — честно скажи:
  «Убытка за период не было, чистая прибыль = X млн руб.». Не выдумывай негатив.

ПОРЯДКИ ВЕЛИЧИН (все суммы в БД — в тыс. руб.):
- > 1 000 000 тыс. → это МИЛЛИАРДЫ:  20 199 392 = 20,2 млрд руб. (не «20 млн»).
- 1 000…1 000 000 тыс. → МИЛЛИОНЫ:   978 270 = 978 млн руб.;  3 451 = 3,5 млн руб.
  −568 171 = убыток 568 млн руб. (нули не теряй: это 568 млн, а не 56,8).
- Масштаб (млн/млрд) указывай явно; можно дать оба: «20,2 млрд руб. (20 199 392 тыс. руб.)».

ЯЗЫК — ТОЛЬКО РУССКИЙ (англицизмы заменяй):
  profitability→рентабельность, liquidity→ликвидность, margin→маржа,
  turnover→оборачиваемость, cash flow→денежный поток, equity→собственный капитал,
  assets→активы, debt→долг/задолженность, revenue→выручка.
  Общепринятые аббревиатуры (РСБУ, EBIT, EBITDA, ROE) можно оставить. Перед
  отправкой проверь: нет ли лишних латинских букв.

ФОРМАТ ОТВЕТА:
- 1–3 строки результата → один абзац с цифрами + фраза о смысле.
- 4+ строк → markdown-список с заголовками «### КОМПАНИЯ» и подпунктами «- ».
- Несколько периодов по одной компании → покажи динамику цепочкой «X → Y → Z»
  и процентное изменение за весь период («снизилось на 38,4%»).
- В конце блок **«Вывод:»** (1–3 фразы): не пересказ цифр, а ДИАГНОЗ ПРИЧИНЫ со
  ссылками на коды строк РСБУ (2110, 2200, 2300, 2340, 2350, 2400 и т.п.).
  Критичные отклонения (убыток, ликвидность <1, резкое падение/рост) назови явно.

КОЭФФИЦИЕНТЫ (только если есть в результате) — нормы:
  Operating Margin 10-15%, Asset Turnover ~1.0, Debtors Share 30-40% (ниже лучше),
  Quick Ratio 1-2, SGA Ratio 10-15% (ниже лучше). Укажи насколько и в какую
  сторону показатель отклонился: «Operating Margin = 8,5% при норме 10-15% —
  ниже отраслевой нормы». Если вопрос не про коэффициенты — раздел игнорируй.

ПРИМЕР (динамика 2022-2025):
### АО ЭКЗ
- **Прочие доходы (стр. 2340):** 598,9 → 459,9 → 378,1 → 229,7 млн руб. — −61,7% за 4 года.
- **Прочие расходы (стр. 2350):** 606,2 → 519,6 → 452,4 → 337,3 млн руб. — −44,4%.

**Вывод:** Прочие доходы (2340) падают быстрее прочих расходов (2350); это вместе
с ростом процентов уплаченных (2330) стало причиной убытка до налогообложения
(2300). Прибыль от продаж (2200) остаётся положительной — проблема во
внеоперационной части ОФР.

Вопрос: {question}

Колонки результата: {columns}
Данные:
{rows}

Краткий ответ:"""


# ============================================================
# SCHEMA INTROSPECTION
# ============================================================

def fetch_schema_text(conn) -> str:
    """
    Делает текстовое DDL-описание схемы finance.* с COMMENT-ами для каждой
    колонки — это и есть «доска для LLM», на которой она пишет SQL.

    Из вывода исключаем таблицу indicator_glossary — её содержимое идёт
    в отдельный раздел промпта через fetch_indicator_glossary(); как
    DDL она LLM ничего не даёт.
    """

    sql = """
    SELECT
        c.table_name,
        c.column_name,
        c.data_type,
        pg_catalog.col_description(
            format('%I.%I', c.table_schema, c.table_name)::regclass::oid,
            c.ordinal_position
        ) AS col_comment
    FROM information_schema.columns c
    WHERE c.table_schema = 'finance'
      AND c.table_name <> 'indicator_glossary'
    ORDER BY c.table_name, c.ordinal_position;
    """

    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    by_table: dict[str, list[tuple[str, str, str | None]]] = {}
    for tbl, col, dtype, comment in rows:
        by_table.setdefault(tbl, []).append((col, dtype, comment))

    parts: list[str] = []
    for tbl, cols in by_table.items():
        parts.append(f"-- finance.{tbl}")
        parts.append(f"CREATE TABLE finance.{tbl} (")
        col_lines = []
        for col, dtype, comment in cols:
            line = f"  {col} {dtype.upper()}"
            if comment:
                line += f"  -- {comment}"
            col_lines.append(line)
        parts.append(",\n".join(col_lines))
        parts.append(");")
        parts.append("")
    return "\n".join(parts)


def fetch_indicator_glossary(conn) -> str:
    """
    Подтягивает справочник коэффициентов из finance.indicator_glossary
    и форматирует как человекочитаемую таблицу для промпта LLM.

    Если таблицы ещё нет (миграция не применена) — возвращает пустую
    строку, чтобы старые установки не ломались.
    """

    sql = """
    SELECT code, display_name, description, formula, norm_range, direction, notes
    FROM finance.indicator_glossary
    ORDER BY code;
    """

    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    except Exception:
        # Таблицы нет — мягко возвращаемся к старому промпту без glossary.
        conn.rollback()
        return ""

    if not rows:
        return ""

    lines: list[str] = []
    lines.append("# Справочник финансовых показателей (finance.indicator_glossary)")
    lines.append("# Это эталонные формулы. Используй их буквально, когда вопрос")
    lines.append("# затрагивает любой из этих коэффициентов.")
    lines.append("")

    for code, display, descr, formula, norm, direction, notes in rows:
        lines.append(f"## {display}  (код: {code})")
        lines.append(f"  Описание: {descr}")
        lines.append(f"  Формула:  {formula}")
        if norm:
            lines.append(f"  Норма:    {norm}  ({direction})")
        else:
            lines.append(f"  Направление: {direction}")
        if notes:
            lines.append(f"  Заметки:  {notes}")
        lines.append("")

    return "\n".join(lines)


def fetch_companies_list(conn) -> list[str]:
    """
    Плоский список short_name всех компаний (для python-safety net'а
    fix_ilike_patterns).

    Если БД недоступна — возвращает пустой список, ничего не падает.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT short_name FROM finance.company ORDER BY short_name;"
            )
            return [row[0] for row in cur.fetchall() if row[0]]
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return []


def fetch_companies_with_segments(conn) -> list[dict]:
    """
    Список компаний с привязкой к отраслевому сегменту.

    Возвращает list[{"name": ..., "segment": ... | None}].
    Если колонка segment ещё не создана (миграция не применена) —
    возвращает имена с segment=None, ничего не падает.

    Сегменты заданы в `finance_segments_migration.sql`. Используется в
    format_companies_for_prompt и в UI шапки/конструктора.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT short_name, segment FROM finance.company "
                "ORDER BY segment NULLS LAST, short_name;"
            )
            return [
                {"name": r[0], "segment": r[1]}
                for r in cur.fetchall() if r[0]
            ]
    except Exception:
        # Колонка segment могла ещё не появиться — мягкий fallback.
        try:
            conn.rollback()
        except Exception:
            pass
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT short_name FROM finance.company "
                    "ORDER BY short_name;"
                )
                return [
                    {"name": r[0], "segment": None}
                    for r in cur.fetchall() if r[0]
                ]
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            return []


def format_companies_for_prompt(companies: list[dict]) -> str:
    """
    Форматирует список компаний как блок для промпта LLM, СГРУППИРОВАННЫЙ
    по сегменту.

    Принимает list[dict] вида {"name": ..., "segment": ... | None}.
    Группировка по сегменту помогает модели сразу понимать состав сегментов
    при вопросах типа «топ-3 кабельщиков по выручке».
    """
    if not companies:
        return ""
    by_segment: dict[str, list[str]] = {}
    for c in companies:
        seg = c.get("segment") or "(сегмент не задан)"
        by_segment.setdefault(seg, []).append(c["name"])

    lines = [
        "# СПИСОК ИЗВЕСТНЫХ КОМПАНИЙ в finance.company "
        "(СГРУППИРОВАНО по segment)",
        "# Это ИСЧЕРПЫВАЮЩИЙ список. Других компаний в БД нет.",
        "# Когда строишь ILIKE-паттерн для WHERE c.short_name ILIKE — бери",
        "# КОРЕНЬ слова из этого списка (без юр. формы и подчёркиваний).",
        "# Не транслитерируй, не «исправляй» написание из вопроса — если",
        "# в вопросе «Билдэкс», а в списке «ООО Билдэкс» — пиши ILIKE",
        "# '%билдэкс%', а НЕ '%билэдэкс%' или '%bildex%'.",
        "# Если вопрос про отрасль/сегмент («кабельщики», «дистры», "
        "«молочка») — фильтруй ПО c.segment, а не перечисляй имена руками.",
        "",
    ]
    # Сортируем сегменты по убыванию числа компаний, NULL-сегмент в конец.
    seg_keys = sorted(
        by_segment.keys(),
        key=lambda s: (s == "(сегмент не задан)", -len(by_segment[s]), s),
    )
    for seg in seg_keys:
        lines.append(f"## Сегмент: {seg}  ({len(by_segment[seg])})")
        for name in by_segment[seg]:
            lines.append(f"  - {name}")
        lines.append("")
    return "\n".join(lines)


_ILIKE_RE = re.compile(r"ILIKE\s+'%([^%']+)%'", re.IGNORECASE)

# Паттерн для поиска фильтров по c.segment: c.segment = 'X', c.segment ILIKE 'X',
# c.segment IN ('X','Y'). Реальный SQL Postgres так же терпит и tbl.segment.
_SEGMENT_FILTER_RE = re.compile(
    r"(?P<col>(?:\w+\.)?segment)\s*"
    r"(?P<op>=|!=|<>|ILIKE|LIKE)\s*"
    r"'(?P<val>[^']+)'",
    re.IGNORECASE,
)


def fetch_segments_list(conn) -> list[str]:
    """Уникальные непустые значения finance.company.segment."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT segment FROM finance.company "
                "WHERE segment IS NOT NULL ORDER BY segment;"
            )
            return [r[0] for r in cur.fetchall() if r[0]]
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return []


def fix_segment_patterns(sql: str, segments: list[str]) -> tuple[str, list[str]]:
    """
    Python safety-net для фильтров по c.segment.

    LLM иногда вместо ТОЧНОГО значения сегмента подставляет обрывок
    имени компании или жаргон («ILIKE '%сибкабель%'», «= 'кабельщики'»).
    Эта функция:
    1. Находит все c.segment <op> 'X' и c.segment ILIKE '%X%' / 'X'.
    2. Если X матчится с реальным сегментом из БД (точно или подстрокой,
       case-insensitive, ё→е, без скобок-уточнений) — оставляет.
    3. Иначе fuzzy через difflib против списка сегментов; при ratio≥0.5
       заменяет на точное значение + оператор = (если был ILIKE).

    Возвращает (исправленный_sql, список_замен).
    """
    if not segments:
        return sql, []

    import difflib

    def _norm(s: str) -> str:
        # Убираем уточнения в скобках, ё→е, lower; так «молочная» сматчится
        # с «Молочная промышленность (раскислитель)».
        s = re.sub(r"\([^)]*\)", "", s).replace("ё", "е")
        return s.lower().strip().strip("%").strip()

    segments_norm = [(s, _norm(s)) for s in segments]
    replacements: list[str] = []

    def _replace(match: re.Match) -> str:
        col = match.group("col")
        op = match.group("op").upper()
        val = match.group("val")
        val_n = _norm(val)

        # Уже совпадает с реальным сегментом (точно или подстрокой)?
        for orig, norm in segments_norm:
            if val_n == norm or val_n in norm or norm in val_n:
                # Если оператор ILIKE, но значение совпадает целиком —
                # лучше переписать как = для строгости.
                if op in ("ILIKE", "LIKE") and val_n == norm:
                    replacements.append(
                        f"{col} {op} '{val}' → {col} = '{orig}' (norm match)"
                    )
                    return f"{col} = '{orig}'"
                # Подстрочное совпадение — конвертируем в точное равенство,
                # если оператор = или ILIKE без шаблонных %.
                if op == "=" and val != orig:
                    replacements.append(
                        f"{col} = '{val}' → {col} = '{orig}' (canonical)"
                    )
                    return f"{col} = '{orig}'"
                if op in ("ILIKE", "LIKE") and "%" not in val:
                    replacements.append(
                        f"{col} {op} '{val}' → {col} = '{orig}'"
                    )
                    return f"{col} = '{orig}'"
                return match.group(0)

        # Не сматчилось — fuzzy
        best, best_r = None, 0.0
        for orig, norm in segments_norm:
            r = difflib.SequenceMatcher(None, val_n, norm).ratio()
            if r > best_r:
                best_r, best = r, orig
        if best and best_r >= 0.5:
            replacements.append(
                f"{col} {op} '{val}' → {col} = '{best}' "
                f"(fuzzy, ratio={best_r:.2f})"
            )
            return f"{col} = '{best}'"

        return match.group(0)

    new_sql = _SEGMENT_FILTER_RE.sub(_replace, sql)
    return new_sql, replacements


def fix_ilike_patterns(sql: str, companies: list[str]) -> tuple[str, list[str]]:
    """
    Python safety-net: после генерации SQL проверяем все ILIKE-паттерны
    для company.short_name. Если ни одна компания не содержит этого
    паттерна как подстроки (нормализованный case + ё→е), пытаемся найти
    ближайшее совпадение через difflib и подменяем паттерн в SQL.

    Возвращает (исправленный_sql, список_замен).
    """
    if not companies:
        return sql, []

    def _norm(s: str) -> str:
        return s.replace("_", " ").replace("ё", "е").lower().strip()

    companies_norm = [(c, _norm(c)) for c in companies]

    import difflib
    replacements: list[str] = []

    def _replace(match: re.Match) -> str:
        pat = match.group(1)
        pat_n = _norm(pat)
        # Если уже матчится хоть с одной компанией — оставляем как есть.
        if any(pat_n in cn for _, cn in companies_norm):
            return match.group(0)

        # Пытаемся найти ближайшую: лучший ratio против короткого
        # «корневого» слова компании (без юр. формы).
        best_company = None
        best_ratio = 0.0
        for orig, norm in companies_norm:
            for word in norm.split():
                if len(word) < 4:
                    continue
                r = difflib.SequenceMatcher(None, pat_n, word).ratio()
                if r > best_ratio:
                    best_ratio = r
                    best_company = word
        if best_company and best_ratio >= 0.6:
            replacements.append(f"'{pat}' → '{best_company}' (ratio={best_ratio:.2f})")
            return f"ILIKE '%{best_company}%'"
        return match.group(0)

    new_sql = _ILIKE_RE.sub(_replace, sql)
    return new_sql, replacements


# ============================================================
# OLLAMA
# ============================================================

def call_ollama(prompt: str, model: str, url: str, timeout: int = 900,
                max_tokens: int = 4096) -> str:
    """
    Базовый вызов LLM для SQL-маршрута.

    Имя «call_ollama» оставлено для backward-compat (вызывается из app.py
    и main() CLI). Внутри — универсальный llm_provider, который
    автоматически выбирает Ollama / Groq / Anthropic / OpenAI по
    переменной окружения LLM_PROVIDER. Аргументы model/url передаются
    в Ollama-режиме; в облачных режимах url игнорируется, model
    используется если задан, иначе подтянется из соответствующих env.

    max_tokens — лимит ответа для облачных провайдеров (Ollama игнорирует).
    Для резюме имеет смысл ставить поменьше (1500), чтобы не съедать TPM.

    Поэтому ничего в коде менять не нужно — переключение локально <->
    cloud делается ОДНОЙ переменной LLM_PROVIDER.
    """
    from llm_provider import call_llm
    return call_llm(prompt, model=model, timeout=timeout, max_tokens=max_tokens)


# ============================================================
# SQL EXTRACTION + VALIDATION
# ============================================================

_FENCE_RE = re.compile(r"```(?:sql)?\s*(.+?)\s*```", re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> str:
    """
    Достаёт SQL из ответа LLM:
    - срезает <think>...</think> блоки (qwen3 может их подмешивать);
    - предпочитает содержимое ```sql ... ```;
    - иначе берёт всё начиная с первого SELECT/WITH.
    """

    text = _THINK_RE.sub("", text).strip()

    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    upper = text.upper()
    # Берём минимальную позицию среди WITH и SELECT — иначе у WITH-запроса
    # «WITH a AS (SELECT 1) SELECT * FROM a» отрезался бы префикс «WITH a AS (».
    candidates = [i for i in (upper.find("WITH"), upper.find("SELECT")) if i >= 0]
    if candidates:
        return text[min(candidates):].strip()

    return text.strip()


_FORBIDDEN_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
    "GRANT", "REVOKE", "CREATE", "MERGE", "REPLACE", "VACUUM",
    "COMMIT", "ROLLBACK", "EXECUTE", "CALL", "DO", "COPY",
}


def validate_sql(sql: str) -> str:
    """
    Нормализует и валидирует SQL. Бросает ValueError при попытке
    что-то модифицирующее или мультистейтментное.
    """

    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise ValueError("Пустой SQL")

    upper = sql.upper().lstrip("(")
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise ValueError(
            f"Разрешён только SELECT/WITH. Получено: {sql[:50]!r}"
        )

    # Несколько statement через ; запрещаем.
    if ";" in sql:
        raise ValueError("Запрещены несколько SQL-выражений (символ ';')")

    # Запрещённые ключевые слова — как отдельные токены.
    tokens = set(re.findall(r"\b[A-Z]+\b", sql.upper()))
    bad = tokens & _FORBIDDEN_KEYWORDS
    if bad:
        raise ValueError(f"Запрещённые ключевые слова в SQL: {sorted(bad)}")

    return sql


def ensure_limit(sql: str, default_limit: int) -> str:
    """Если в запросе нет LIMIT — дописываем."""
    if re.search(r"\bLIMIT\s+\d+", sql, flags=re.IGNORECASE):
        return sql
    return sql + f"\nLIMIT {default_limit}"


# ============================================================
# EXECUTE
# ============================================================

def execute_sql(conn, sql: str, timeout_ms: int) -> tuple[list[str], list[tuple]]:
    """
    Исполняет SQL в read-only транзакции с жёстким таймаутом.
    Возвращает (columns, rows).
    """

    with conn.cursor() as cur:
        cur.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute(sql)
        columns = [d.name for d in cur.description] if cur.description else []
        rows = cur.fetchall()
    return columns, rows


# ============================================================
# DISPLAY
# ============================================================

def _fmt_cell(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}".replace(",", " ")
        return f"{value:,.2f}".replace(",", " ")
    if isinstance(value, int):
        return f"{value:,}".replace(",", " ")
    return str(value)


def print_table(columns: list[str], rows: list[tuple], max_rows: int = 200) -> None:
    if not columns:
        print("(нет колонок в результате)")
        return
    if not rows:
        print("(пустой результат, 0 строк)")
        return

    cells = [[_fmt_cell(v) for v in row] for row in rows[:max_rows]]
    widths = [len(c) for c in columns]
    for row in cells:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))

    sep_line = "-+-".join("-" * w for w in widths)
    print(" | ".join(c.ljust(widths[i]) for i, c in enumerate(columns)))
    print(sep_line)
    for row in cells:
        print(" | ".join(v.ljust(widths[i]) for i, v in enumerate(row)))

    if len(rows) > max_rows:
        print(f"... (показано {max_rows} из {len(rows)} строк)")
    else:
        print(f"\n({len(rows)} строк)")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Text-to-SQL по схеме finance.*")
    parser.add_argument("question", type=str, help="Вопрос на русском")
    parser.add_argument(
        "--model",
        default=None,
        help="Модель Ollama. По умолчанию OLLAMA_MODEL_SQL → OLLAMA_MODEL → qwen3:14b",
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="URL Ollama /api/generate. По умолчанию OLLAMA_URL из .env",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Авто-LIMIT, если LLM забудет. Default: {DEFAULT_LIMIT}",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help=f"statement_timeout в миллисекундах. Default: {DEFAULT_TIMEOUT_MS}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Сгенерить SQL и распечатать, но не исполнять.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="После результата LLM кратко резюмирует данные на русском.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Печатать промпт, сырой ответ LLM и схему.",
    )
    return parser.parse_args()


def get_conn():
    import psycopg2

    return psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
    )


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    model = (args.model
             or os.getenv("OLLAMA_MODEL_SQL")
             or os.getenv("OLLAMA_MODEL")
             or DEFAULT_MODEL)
    ollama_url = (args.ollama_url
                  or os.getenv("OLLAMA_URL")
                  or DEFAULT_OLLAMA_URL)

    print("=" * 78)
    print(f"Question: {args.question}")
    print(f"Model:    {model}")
    print(f"Limit:    {args.limit}, timeout {args.timeout_ms} ms")
    print("=" * 78)

    conn = get_conn()
    try:
        # 1) Схема + справочник коэффициентов + промпт
        schema_text = fetch_schema_text(conn)
        glossary_text = fetch_indicator_glossary(conn)
        companies_rich = fetch_companies_with_segments(conn)
        companies_list = [c["name"] for c in companies_rich]
        companies_text = format_companies_for_prompt(companies_rich)

        if args.debug:
            print("\n--- SCHEMA ---")
            print(schema_text)
            if glossary_text:
                print("\n--- GLOSSARY ---")
                print(glossary_text)
            if companies_text:
                print("\n--- COMPANIES ---")
                print(companies_text)

        prompt = PROMPT_TEMPLATE.format(
            schema=schema_text,
            companies=companies_text,
            glossary=glossary_text,
            examples=format_examples(),
            limit=args.limit,
            question=args.question,
        )
        if args.debug:
            print(f"\n--- PROMPT ({len(prompt)} chars) ---")
            print(prompt[:2000])
            print("...")

        # 2) Ollama -> SQL
        print("\n[ollama] генерирую SQL...")
        raw = call_ollama(prompt, model=model, url=ollama_url)
        if args.debug:
            print(f"\n--- RAW LLM RESPONSE ---\n{raw}")

        sql = extract_sql(raw)
        sql = validate_sql(sql)
        sql = ensure_limit(sql, args.limit)
        # Safety-nets: исправляем галлюцинированные имена компаний и
        # «странные» фильтры по сегменту.
        sql, _ = fix_ilike_patterns(sql, companies_list)
        sql, _ = fix_segment_patterns(sql, fetch_segments_list(conn))

        print()
        print("=" * 78)
        print("SQL:")
        print("=" * 78)
        print(sql)

        if args.dry_run:
            print("\n(dry-run: не исполняем)")
            return

        # 3) Исполнение
        print()
        print("=" * 78)
        print("RESULT:")
        print("=" * 78)
        columns, rows = execute_sql(conn, sql, args.timeout_ms)
        print_table(columns, rows)
        conn.rollback()  # выходим из read-only транзакции чисто

        # 4) Опциональное резюме LLM
        if args.explain and rows:
            print()
            print("[ollama] резюмирую...")
            summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
                question=args.question,
                columns=", ".join(columns),
                rows="\n".join(
                    " | ".join(_fmt_cell(v) for v in r)
                    for r in rows[:50]
                ),
            )
            summary = call_ollama(summary_prompt, model=model, url=ollama_url)
            print()
            print("=" * 78)
            print("РЕЗЮМЕ:")
            print("=" * 78)
            print(summary)

    except ValueError as exc:
        print(f"\n[VALIDATION ERROR] {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
