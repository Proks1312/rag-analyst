"""
ETL: загрузка финансовой отчётности из xlsx (Контур.Фокус) в Postgres.

Читает книги Excel из data/raw/financial excels (rus)/, парсит секции
«Форма №1» (баланс) и «Форма №2» (ОФР), маппит коды строк РСБУ на
колонки схемы finance.* и делает UPSERT.

Запуск:
    # Сначала применить DDL (один раз):
    Get-Content finance_schema.sql | docker exec -i rag_postgres psql -U rag_user -d rag_db

    # Сухой прогон — посмотреть, что бы загрузилось, без записи в БД:
    python load_finance_xlsx_to_db.py --dry-run

    # Реальная загрузка:
    python load_finance_xlsx_to_db.py
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import date
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# CODE -> COLUMN MAPPING (РСБУ)
# ============================================================

# Форма №1. Дубли (1100 «Итого», 1150 «Основные средства»/«Материальные
# внеоборотные активы» и т.п. — это варианты Контур.Фокуса для одного и
# того же кода. Маппим по коду, имя в самой строке игнорируем.
BALANCE_CODE_MAP: dict[str, str] = {
    # Внеоборотные
    "1110": "c_1110_nma",
    "1120": "c_1120_niokr",
    "1130": "c_1130_npa",
    "1140": "c_1140_mpa",
    "1150": "c_1150_os",
    "1160": "c_1160_dvmc",
    "1170": "c_1170_finvloj_dolg",
    "1180": "c_1180_ona",
    "1190": "c_1190_proch_vneobor",
    "1100": "c_1100_vneobor_total",
    # Оборотные
    "1210": "c_1210_zapasy",
    "1220": "c_1220_nds",
    "1230": "c_1230_debit",
    "1240": "c_1240_finvloj_kratk",
    "1250": "c_1250_dengi",
    "1260": "c_1260_proch_oborot",
    "1200": "c_1200_oborot_total",
    # Капитал
    "1310": "c_1310_ustav_kap",
    "1320": "c_1320_sobst_akcii",
    "1340": "c_1340_pereocenka",
    "1350": "c_1350_dobav_kap",
    "1360": "c_1360_rezerv_kap",
    "1370": "c_1370_neraspr_pribyl",
    "1300": "c_1300_kapital_total",
    # Долгосрочные
    "1410": "c_1410_dolg_zaem",
    "1420": "c_1420_otloj_nalog_obyaz",
    "1430": "c_1430_ocenoch_obyaz_dolg",
    "1450": "c_1450_proch_dolg",
    "1400": "c_1400_dolgosroch_total",
    # Краткосрочные
    "1510": "c_1510_kratk_zaem",
    "1520": "c_1520_kred_zadolj",
    "1530": "c_1530_dohody_bud_per",
    "1540": "c_1540_ocenoch_obyaz_kratk",
    "1550": "c_1550_proch_kratk",
    "1500": "c_1500_kratkosroch_total",
    # Баланс
    "1600": "c_1600_balans",
}

# Форма №2.
INCOME_CODE_MAP: dict[str, str] = {
    "2110": "c_2110_vyruchka",
    "2120": "c_2120_sebest",
    "2100": "c_2100_valovaya_pribyl",
    "2210": "c_2210_komm_rashody",
    "2220": "c_2220_uprav_rashody",
    "2200": "c_2200_pribyl_prodazh",
    "2310": "c_2310_dohody_ucastiya",
    "2320": "c_2320_proc_polychit",
    "2330": "c_2330_proc_uplat",
    "2340": "c_2340_proch_dohody",
    "2350": "c_2350_proch_rashody",
    "2300": "c_2300_pribyl_do_nalog",
    "2410": "c_2410_nalog_pribyl",
    "2411": "c_2411_tek_nalog",
    "2412": "c_2412_otloj_nalog",
    "2421": "c_2421_post_nalog_obyaz",
    "2430": "c_2430_izmen_otloj_obyaz",
    "2450": "c_2450_izmen_otloj_akt",
    "2460": "c_2460_proch_ofr",
    "2400": "c_2400_chistaya_pribyl",
    "2500": "c_2500_sovokup_finrez",
    "2510": "c_2510_rez_pereocenki",
    "2520": "c_2520_rez_proch_oper",
    "2530": "c_2530_nalog_oper",
    "2900": "c_2900_baz_pribyl_akcii",
    "2910": "c_2910_razv_pribyl_akcii",
}


# ============================================================
# XLSX PARSER
# ============================================================

def parse_xlsx(path: Path) -> dict:
    """
    Парсит одну книгу. Возвращает структуру:
        {
          "Форма №1": {year (int): {"Начало": {code: val, ...},
                                     "Конец":  {code: val, ...}}},
          "Форма №2": {year (int): {"Начало": {...}, "Конец": {...}}},
          ...
        }
    Только формы 1 и 2 — остальные секции пропускаем (v1 схемы).
    """

    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook.worksheets[0]
        rows = list(worksheet.iter_rows(values_only=True))
    finally:
        workbook.close()

    result: dict[str, dict] = {}
    current_form: str | None = None
    year_at_col: dict[int, int] = {}        # col -> year
    period_at_col: dict[int, str] = {}      # col -> "Начало"/"Конец"

    for row_idx, row in enumerate(rows):
        if not row:
            continue
        c0 = _cell_str(row[0])
        c1 = _cell_str(row[1] if len(row) > 1 else "")

        # Новая секция «Форма №N» — сбрасываем разметку.
        if c0.startswith("Форма "):
            current_form = c0
            year_at_col = {}
            period_at_col = {}
            # Годы — в этой же строке, в колонках 2,4,6,8...
            for j in range(2, len(row), 2):
                yc = _cell_str(row[j])
                m = re.match(r"^(20\d{2})", yc)
                if m:
                    year = int(m.group(1))
                    year_at_col[j] = year       # Начало
                    year_at_col[j + 1] = year   # Конец
            continue

        # Строка с заголовками периодов: «Код | Начало | Конец | …».
        # Идёт сразу после «Форма №N»; распознаём по наличию «Начало»/«Конец».
        if current_form and not period_at_col:
            cells_lower = [(_cell_str(c).lower(), j) for j, c in enumerate(row)]
            if any(c == "начало" or c == "конец" for c, _ in cells_lower):
                for txt, j in cells_lower:
                    if txt == "начало":
                        period_at_col[j] = "Начало"
                    elif txt == "конец":
                        period_at_col[j] = "Конец"
                continue

        # Строка данных — в колонке 1 код из 4 цифр.
        if current_form and c1 and re.fullmatch(r"\d{4}", c1):
            code = c1
            form_dict = result.setdefault(current_form, {})
            for col_idx, year in year_at_col.items():
                if col_idx >= len(row):
                    continue
                period = period_at_col.get(col_idx)
                if period is None:
                    continue
                val = _cell_num(row[col_idx])
                if val is None:
                    continue
                year_dict = form_dict.setdefault(year, {})
                period_dict = year_dict.setdefault(period, {})
                # Если код повторяется (дубликаты вариантов имени) —
                # последнее непустое значение выигрывает.
                period_dict[code] = val

    return result


def _cell_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _cell_num(value):
    """Превращает значение ячейки в число (или None)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Целые float (например 7976290.0) — приводим к int.
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    s = str(value).strip().replace("\xa0", "").replace(" ", "")
    if not s:
        return None
    s = s.replace(",", ".")
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return None


# ============================================================
# DB HELPERS
# ============================================================

def get_conn():
    # Ленивый импорт — чтобы dry-run работал даже без psycopg2 в venv.
    import psycopg2

    load_dotenv(PROJECT_ROOT / ".env")
    return psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
    )


def upsert_company(conn, short_name: str) -> int:
    """
    UPSERT компании по short_name; возвращает её id.
    full_name/ogrn/inn оставляем NULL — заполним позже из PDF.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO finance.company (short_name)
            VALUES (%s)
            ON CONFLICT (short_name) DO UPDATE SET short_name = EXCLUDED.short_name
            RETURNING id;
            """,
            (short_name,),
        )
        return cur.fetchone()[0]


def upsert_balance_row(conn, company_id: int, year: int, period_date: date,
                       values: dict[str, float | int]) -> None:
    cols = ["company_id", "reporting_year", "period_date"] + list(values.keys())
    placeholders = ", ".join(["%s"] * len(cols))
    update_assignments = ", ".join(f"{k} = EXCLUDED.{k}" for k in values.keys())
    sql = f"""
        INSERT INTO finance.balance_sheet ({', '.join(cols)})
        VALUES ({placeholders})
        ON CONFLICT (company_id, reporting_year, period_date)
        DO UPDATE SET {update_assignments};
    """
    params = [company_id, year, period_date] + list(values.values())
    with conn.cursor() as cur:
        cur.execute(sql, params)


def upsert_income_row(conn, company_id: int, year: int,
                      values: dict[str, float | int]) -> None:
    cols = ["company_id", "reporting_year"] + list(values.keys())
    placeholders = ", ".join(["%s"] * len(cols))
    update_assignments = ", ".join(f"{k} = EXCLUDED.{k}" for k in values.keys())
    sql = f"""
        INSERT INTO finance.income_statement ({', '.join(cols)})
        VALUES ({placeholders})
        ON CONFLICT (company_id, reporting_year)
        DO UPDATE SET {update_assignments};
    """
    params = [company_id, year] + list(values.values())
    with conn.cursor() as cur:
        cur.execute(sql, params)


# ============================================================
# SEGMENTS (company.segment <- finance.segment_map)
# ============================================================

# Применяет справочник finance.segment_map к company.segment.
# На каждую компанию берётся самый длинный (самый специфичный) подошедший
# ILIKE-паттерн — это исключает неоднозначность при пересечении шаблонов.
# Тот же запрос лежит в finance_segment_map.sql (секция 4).
APPLY_SEGMENTS_SQL = """
UPDATE finance.company c
SET segment = sub.segment
FROM (
    SELECT DISTINCT ON (comp_id) comp_id, segment
    FROM (
        SELECT c.id AS comp_id, m.segment, length(m.pattern) AS plen
        FROM finance.company c
        JOIN finance.segment_map m ON c.short_name ILIKE m.pattern
    ) j
    ORDER BY comp_id, plen DESC
) sub
WHERE c.id = sub.comp_id;
"""


def ensure_segment_schema(conn) -> None:
    """
    Идемпотентно гарантирует, что колонка company.segment и таблица
    finance.segment_map существуют. Сами строки маппинга НЕ сидим здесь —
    источник истины это finance_segment_map.sql / правки в Adminer.
    """
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE finance.company ADD COLUMN IF NOT EXISTS segment TEXT;")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_company_segment ON finance.company(segment);"
        )
        # Без PRIMARY KEY на pattern: дефолтная коллация БД склеивает близкие
        # строки (ё=е, _=пробел) и роняет ON CONFLICT. Уникальность — отдельным
        # байт-точным индексом (COLLATE "C"), колонка остаётся в дефолтной
        # коллации, чтобы ILIKE корректно сворачивал регистр кириллицы.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS finance.segment_map (
                pattern  TEXT NOT NULL,
                segment  TEXT NOT NULL
            );
            """
        )
        cur.execute(
            'CREATE UNIQUE INDEX IF NOT EXISTS uq_segment_map_pattern '
            'ON finance.segment_map (pattern COLLATE "C");'
        )
    conn.commit()


def apply_segments(conn) -> list[str]:
    """
    Проставляет company.segment из finance.segment_map и возвращает список
    short_name компаний, для которых сегмент так и не определился (NULL) —
    их нужно добавить в segment_map.
    """
    with conn.cursor() as cur:
        cur.execute(APPLY_SEGMENTS_SQL)
        cur.execute(
            "SELECT short_name FROM finance.company "
            "WHERE segment IS NULL ORDER BY short_name;"
        )
        uncovered = [r[0] for r in cur.fetchall()]
    conn.commit()
    return uncovered


# ============================================================
# FILE -> ROWS
# ============================================================

def short_name_from_filename(path: Path) -> str:
    """
    Извлекает имя компании из имени файла xlsx.
    «АО_СИБКАБЕЛЬ.xlsx» -> «АО СИБКАБЕЛЬ».
    Подчёркивания заменяем на пробел для удобства поиска в SQL по ILIKE.
    """
    stem = path.stem
    return stem.replace("_", " ").strip()


def build_balance_rows(parsed: dict) -> list[tuple[int, date, dict]]:
    """
    Из распарсенного xlsx делает список (year, period_date, {col: val}) для
    finance.balance_sheet.

    period_date: Начало -> 1 января, Конец -> 31 декабря.
    """
    out: list[tuple[int, date, dict]] = []
    form1 = parsed.get("Форма №1", {})
    for year, periods in form1.items():
        for period_label, code_to_val in periods.items():
            period_date = date(year, 1, 1) if period_label == "Начало" else date(year, 12, 31)
            mapped: dict[str, float | int] = {}
            for code, val in code_to_val.items():
                col = BALANCE_CODE_MAP.get(code)
                if col:
                    mapped[col] = val
            if mapped:
                out.append((year, period_date, mapped))
    return out


def build_income_rows(parsed: dict) -> list[tuple[int, dict]]:
    """
    Из распарсенного xlsx делает список (year, {col: val}) для
    finance.income_statement. Берём «Конец» как годовой итог;
    если «Конец» пуст для года, fallback на «Начало».
    """
    out: list[tuple[int, dict]] = []
    form2 = parsed.get("Форма №2", {})
    for year, periods in form2.items():
        source = periods.get("Конец") or periods.get("Начало") or {}
        mapped: dict[str, float | int] = {}
        for code, val in source.items():
            col = INCOME_CODE_MAP.get(code)
            if col:
                mapped[col] = val
        if mapped:
            out.append((year, mapped))
    return out


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Load РСБУ xlsx (Контур.Фокус) into finance.* schema."
    )
    p.add_argument(
        "--xlsx-dir",
        default="data/raw/financial excels (rus)",
        help="Folder with *.xlsx files",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Только распарсить и напечатать суммы, без записи в БД.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    xlsx_dir = Path(args.xlsx_dir)
    if not xlsx_dir.is_absolute():
        xlsx_dir = PROJECT_ROOT / xlsx_dir

    # Пропускаем временные lock-файлы Excel (~$Имя.xlsx), иначе openpyxl
    # падает на них с BadZipFile и роняет весь прогон.
    files = [
        f for f in sorted(xlsx_dir.glob("*.xlsx"))
        if not f.name.startswith("~$")
    ]
    if not files:
        print(f"Не нашёл xlsx в {xlsx_dir}")
        return

    print("=" * 78)
    print("LOAD FINANCE XLSX -> DB")
    print("=" * 78)
    print(f"Source dir: {xlsx_dir}")
    print(f"Файлов:     {len(files)}")
    print(f"Dry run:    {args.dry_run}")
    print("=" * 78)

    conn = None if args.dry_run else get_conn()
    if conn is not None:
        ensure_segment_schema(conn)

    try:
        total_balance_rows = 0
        total_income_rows = 0

        for fp in files:
            short_name = short_name_from_filename(fp)
            parsed = parse_xlsx(fp)
            balance_rows = build_balance_rows(parsed)
            income_rows = build_income_rows(parsed)

            print(
                f"  {short_name:42s}  "
                f"balance: {len(balance_rows):>3} строк, "
                f"income: {len(income_rows):>2} строк"
            )

            total_balance_rows += len(balance_rows)
            total_income_rows += len(income_rows)

            if args.dry_run:
                continue

            company_id = upsert_company(conn, short_name)
            for year, period_date, values in balance_rows:
                upsert_balance_row(conn, company_id, year, period_date, values)
            for year, values in income_rows:
                upsert_income_row(conn, company_id, year, values)
            conn.commit()

        # Проставляем сегменты из справочника finance.segment_map.
        uncovered: list[str] = []
        if conn is not None:
            uncovered = apply_segments(conn)

        print("=" * 78)
        print("DONE")
        print("=" * 78)
        print(f"balance_sheet rows:     {total_balance_rows}")
        print(f"income_statement rows:  {total_income_rows}")
        if args.dry_run:
            print("(dry-run: в БД ничего не записано)")
        elif uncovered:
            print("=" * 78)
            print(f"⚠️  Без сегмента ({len(uncovered)}) — добавь паттерн в")
            print("    finance.segment_map (Adminer) или в finance_segment_map.sql:")
            for name in uncovered:
                print(f"      • {name}")
        else:
            print("Сегменты: все компании покрыты finance.segment_map.")
        print("=" * 78)

    except Exception:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
