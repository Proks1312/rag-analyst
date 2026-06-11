r"""
Streamlit-фронт для аналитика. Одно поле ввода, два маршрута:

  RAG  -> answer_question.py: vector search по документам + LLM-ответ.
  SQL  -> sql_question.py:    LLM пишет SELECT по схеме finance.*, Postgres исполняет.

По умолчанию маршрут выбирает роутер (по эвристикам в вопросе),
но в сайдбаре можно форсировать вручную.

Запуск:
    .\.venv\Scripts\python.exe -m streamlit run app.py

Зависимости: streamlit, pandas, psycopg2, requests. Для локальной разработки
ещё python-dotenv (на Streamlit Cloud не обязателен).
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd

# python-dotenv — нужен только локально (читает .env). На Streamlit Cloud
# секреты идут через st.secrets, dotenv не обязателен.
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):  # noqa: D401
        """Stub: на Streamlit Cloud python-dotenv может быть не установлен."""
        return False


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Локально подтянет .env. На Streamlit Cloud файла нет — stub вернёт False.
load_dotenv(PROJECT_ROOT / ".env")


def _bootstrap_secrets_to_env() -> None:
    """
    Подкладываем Streamlit Secrets в os.environ.

    Локально (запуск через `streamlit run app.py` на твоей машине) —
    конфигурация читается из .env через load_dotenv выше. В этом случае
    st.secrets — пустой, и функция тихо ничего не делает.

    На Streamlit Cloud — .env нет, зато ты заполняешь Secrets в UI
    Streamlit Cloud в TOML-формате. Этот код копирует их в os.environ
    при старте процесса, чтобы существующий код (который везде
    читает os.getenv(...)) работал без правок.
    """
    try:
        # st.secrets — это специальный AttrDict, итерируется как dict.
        # Внутри Streamlit Cloud он подтягивается из секретов проекта.
        secrets_dict = dict(st.secrets)
    except Exception:
        # Никаких секретов — это локальная разработка без secrets.toml.
        return

    for key, value in secrets_dict.items():
        # Заливаем только строковые секреты (TOML-объекты не подкладываем).
        if isinstance(value, str) and key not in os.environ:
            os.environ[key] = value


_bootstrap_secrets_to_env()


# ============================================================
# ROUTER
# ============================================================

# Регулярные выражения и словари для классификации вопроса.
# Намеренно простые — чтобы поведение роутера было предсказуемым.

_COMPANY_RE = re.compile(
    r"\b(АО|ООО|ПАО|ЗАО|ОАО|НПАО|ОАО|ИП)[_ \-А-ЯЁA-Z][А-ЯЁA-Z_ \-]+",
    re.UNICODE,
)
_YEAR_RE = re.compile(r"\b20\d{2}\b")

# Финансовые метрики и связанные термины — если они есть в вопросе,
# вопрос почти наверняка про точные цифры из finance.*.
_FINANCE_TRIGGERS = (
    "выручк", "прибыл", "убыток", "убытк",
    "баланс", "валюта баланса", "офр",
    "капитал", "обязательств", "актив", "пассив",
    "себестоимост", "доход", "расход",
    "налог", "дебитор", "кредитор",
    "тыс. руб", "млн руб", "млрд руб",
    "топ-", "топ ",
    "сравни",
    "динамик",  # «динамика выручки» — явный SQL-запрос
    "рентабельност", "ликвидност", "оборачиваемост",
    "маржа", "маржинальност",
    "коэффициент", "показатель",
    "банкротств",  # «вероятность банкротства» — SQL по quick_ratio и т.п.
    "quick ratio", "operating margin", "asset turnover",
    "net margin", "debtors share", "sga ratio",
    # Аналитические запросы про общее финансовое состояние компании
    "финансов",         # «финансовое здоровье», «финансовые показатели»
    "оцени", "оценишь", "оценка",
    "здоров", "состояни",  # «состояние компании»
    "положени",         # «положение компании»
)

# Маркеры, которые ОЧЕВИДНО за RAG (качественные вопросы).
_RAG_TRIGGERS = (
    "что в отчёте", "что в отчете", "о чём", "о чем",
    "расскажи про", "опиши", "что говорит",
    "какие риски", "какие выводы",
    "что такое", "что означает",
)

# Follow-up маркеры. Если они есть, и при этом нет жёстких RAG/SQL
# триггеров, маршрут наследуется из последнего assistant-сообщения.
# "Структурируй своё резюме" — структурируй.
# "А за 2024?" — а за.
# "Сократи" / "разверни" / "поясни" — производные форматирования.
_FOLLOWUP_TRIGGERS = (
    # Просьбы переформатировать предыдущий ответ
    "структурируй", "переформатируй", "оформи списком", "оформи как",
    "сделай списком", "сделай таблицей",
    "сократи", "поясни", "разверни", "уточни", "перепиши",
    "переведи", "продолжи", "добавь",
    # Уточняющие вопросы
    "а за ", "а у ", "а в ", "а по ",
    "а какая", "а какой", "а какие", "а каково",
    "а сколько", "а кто", "а что у",
    "то же ", "также для", "и ещё", "ещё для",
)


def _last_assistant_route():
    """Маршрут последнего assistant-сообщения из чат-истории."""
    import streamlit as st
    msgs = st.session_state.get("chat_messages", []) or []
    for msg in reversed(msgs):
        if msg.get("role") == "assistant" and msg.get("route"):
            return msg["route"]
    return None


# Имена компаний из finance.company, нормализованные для поиска без
# префикса (АО/ООО/ПАО). Кэшируем в module-level переменной — список
# из БД редко меняется и не должен дёргать БД при каждом вопросе.
_COMPANY_NAMES_CACHE: list[tuple[str, str]] | None = None


def _get_company_names_cache() -> list[tuple[str, str]]:
    """
    Подтягивает имена из finance.company, делает пары (full_short, core),
    где core — имя без префикса для поиска в свободном тексте.

    Пример: «АО_СИБКАБЕЛЬ» → core «сибкабель» (lower, без подчёркиваний).
    Кэшируется один раз за процесс Streamlit.
    """
    global _COMPANY_NAMES_CACHE
    if _COMPANY_NAMES_CACHE is not None:
        return _COMPANY_NAMES_CACHE

    try:
        names = _list_companies()
    except Exception:
        _COMPANY_NAMES_CACHE = []
        return _COMPANY_NAMES_CACHE

    pairs: list[tuple[str, str]] = []
    for n in names or []:
        # Убираем префикс по списку известных юр-форм.
        core = n
        for prefix in ("АО_", "АО ", "ООО_", "ООО ", "ПАО_", "ПАО ",
                       "ЗАО_", "ЗАО ", "ОАО_", "ОАО ", "НПАО_", "НПАО ",
                       "ИП_", "ИП ", "КРТ_"):
            if core.upper().startswith(prefix):
                core = core[len(prefix):]
                break
        # «ОАО_Курскрезинотехника» уже без префикса в КРТ-варианте.
        core = core.replace("_", " ").strip().lower()
        # 3 буквы (ЭКЗ, НЛМК — это 4) — допустимо, иначе короткие аббревиатуры
        # типа «ЭКЗ» теряются. Слова <3 букв всё-таки пропускаем.
        if len(core) >= 3:
            pairs.append((n, core))

    _COMPANY_NAMES_CACHE = pairs
    return _COMPANY_NAMES_CACHE


def _company_in_question(question: str) -> str | None:
    """
    Ищет имя компании из БД в вопросе без префикса юр. формы.

    Возвращает short_name из finance.company или None.
    Пример: «динамика выручки у Сибкабеля» → «АО_СИБКАБЕЛЬ».
    """
    q_lower = question.lower()
    for short_name, core in _get_company_names_cache():
        if core in q_lower:
            return short_name
    return None


def route(question: str, override: str) -> tuple[str, str]:
    """
    Возвращает (chosen_route, reason).

    В текущей продакшен-конфигурации RAG-маршрут ОТКЛЮЧЁН (документная
    часть требует доработки). Поэтому всегда возвращаем SQL. Код роутера
    с распознаванием company/year/finance триггеров остаётся ниже как
    «информационный» — он определяет понятную причину для UI.
    """
    q_lower = question.lower()

    has_company = bool(_COMPANY_RE.search(question))
    has_year = bool(_YEAR_RE.search(question))
    has_finance = any(t in q_lower for t in _FINANCE_TRIGGERS)
    company_in_dict = _company_in_question(question)
    if company_in_dict:
        has_company = True

    bits = []
    if has_company:
        bits.append(f"компания «{company_in_dict}»" if company_in_dict else "компания")
    if has_year:
        bits.append("год")
    if has_finance:
        bits.append("фин. показатель")

    # Follow-up маркеры — учитываем как причину
    followup_hit = next((t for t in _FOLLOWUP_TRIGGERS if t in q_lower), None)
    if followup_hit:
        bits.append(f"follow-up: «{followup_hit.strip()}»")

    reason = "в вопросе: " + " + ".join(bits) if bits else "финансовый запрос (default)"
    return "SQL", reason


# ============================================================
# SQL PATH (использует sql_question.py)
# ============================================================

def run_sql(question: str, model: str, ollama_url: str,
            limit: int = 100, timeout_ms: int = 10_000) -> dict:
    import psycopg2
    import sql_question as sq

    conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
    )
    try:
        schema = sq.fetch_schema_text(conn)
        glossary = sq.fetch_indicator_glossary(conn)
        # Подмешиваем точный список компаний (сгруппированный по сегменту) —
        # у Llama-4 без него часть имён пишется с галлюцинированной
        # транслитерацией («Билдэкс» → «билэдэкс»), и Postgres не сматчивает.
        # Сегмент в списке нужен, чтобы LLM мог сразу выбирать через
        # c.segment ILIKE ... для вопросов про отрасли.
        companies_rich = sq.fetch_companies_with_segments(conn)
        companies = [c["name"] for c in companies_rich]
        companies_block = sq.format_companies_for_prompt(companies_rich)
        prompt = sq.PROMPT_TEMPLATE.format(
            schema=schema,
            companies=companies_block,
            glossary=glossary,
            examples=sq.format_examples(),
            limit=limit,
            question=question,
        )

        t0 = time.time()
        raw = sq.call_ollama(prompt, model=model, url=ollama_url)
        sql = sq.extract_sql(raw)
        sql = sq.validate_sql(sql)
        sql = sq.ensure_limit(sql, limit)
        # Python safety-net (2 слоя):
        # 1) ILIKE по c.short_name — если LLM сгаллюцинировала имя,
        #    fuzzy-сматчиваем против реальных компаний.
        # 2) Фильтр по c.segment — если LLM подставила обрывок имени
        #    компании или жаргон, заменяем на точное значение из БД.
        sql, ilike_fixes = sq.fix_ilike_patterns(sql, companies)
        segments_list = sq.fetch_segments_list(conn)
        sql, segment_fixes = sq.fix_segment_patterns(sql, segments_list)
        t_gen = time.time() - t0

        t0 = time.time()
        columns, rows = sq.execute_sql(conn, sql, timeout_ms)
        conn.rollback()  # закрываем READ ONLY-транзакцию
        t_exec = time.time() - t0

        # Дедупликация: иногда LLM пишет JOIN income_statement + balance_sheet
        # без фильтра по period_date, и каждый год удваивается. Если видим
        # точные дубли строк (вся строка совпадает) — оставляем уникальные.
        rows, was_deduped = _dedup_rows(rows)

        return {
            "ok": True,
            "sql": sql,
            "columns": columns,
            "rows": rows,
            "raw": raw,
            "time_generate": t_gen,
            "time_execute": t_exec,
            "deduped": was_deduped,
        }
    finally:
        conn.close()


def _dedup_rows(rows: list[tuple]) -> tuple[list[tuple], bool]:
    """
    Убирает точные дубликаты строк, сохраняя порядок появления.
    Возвращает (deduped_rows, was_deduped).
    """
    if not rows:
        return rows, False
    seen = set()
    out: list[tuple] = []
    for r in rows:
        # tuple из row делаем хешируемым (Decimal/None/прочее уже OK).
        try:
            key = tuple(r)
        except TypeError:
            # Если в строке есть нехешируемые объекты — отказываемся от дедупа.
            return rows, False
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out, len(out) < len(rows)


def run_sql_summary(question: str, columns: list[str], rows: list[tuple],
                    model: str, ollama_url: str) -> str:
    """
    Готовит rows для SUMMARY-промпта в человекочитаемом виде.

    Раньше отдавали сырые числа («601770»), и модели уровня glm4:9b
    путались со знаками и порядками. Теперь каждое число в строке
    предобрабатываем: "601 770 (= 601,8 млн руб)". Так модель видит
    и сырое значение, и подсказку какого оно порядка, и явный знак.
    """
    import sql_question as sq
    import math
    from decimal import Decimal

    def _fmt_summary_value(val, col_name: str) -> str:
        if val is None:
            return "NULL"
        col_low = col_name.lower()
        is_year = (
            "year" in col_low or "год" in col_low or "period" in col_low
        )
        if not isinstance(val, (int, float, Decimal)):
            return str(val)
        try:
            f = float(val)
        except (TypeError, ValueError):
            return str(val)
        if math.isnan(f):
            return "NaN"
        if is_year:
            return str(int(f))

        abs_f = abs(f)
        sign = "-" if f < 0 else ""
        # Целое — показываем как есть (например, для коэффициентов и пр.).
        if abs_f != int(abs_f):
            # Дробное (коэффициенты: 0,75 или 1,84)
            return f"{f:.2f}".replace(".", ",")

        n_int = int(abs_f)
        raw = f"{sign}{n_int:,}".replace(",", " ")
        if abs_f < 1_000:
            return raw
        if abs_f < 1_000_000:
            return f"{raw} (= {sign}{abs_f/1000:.1f} млн руб)".replace(".", ",")
        return f"{raw} (= {sign}{abs_f/1_000_000:.2f} млрд руб)".replace(".", ",")

    formatted_rows = []
    for r in rows[:50]:
        parts = []
        for col, v in zip(columns, r):
            parts.append(f"{col}={_fmt_summary_value(v, col)}")
        formatted_rows.append(" | ".join(parts))
    rows_text = "\n".join(formatted_rows)

    prompt = sq.SUMMARY_PROMPT_TEMPLATE.format(
        question=question,
        columns=", ".join(columns),
        rows=rows_text,
    )

    # На cloud-провайдерах резюме катим на ЛЁГКОЙ модели и с меньшим
    # max_tokens. Причина: Groq free-tier имеет жёсткий TPM-лимит на 70b
    # versatile (12k/min), и пара «тяжёлый SQL + длинный SUMMARY-промпт»
    # упирается в него — резюме падает с 429 rate_limit_exceeded. У 8b-instant
    # лимит ~30k/min, плюс модель ощутимо быстрее. На локальной Ollama же
    # ничего не меняется (там лимитов нет).
    provider = (os.getenv("LLM_PROVIDER") or "ollama").lower().strip()
    if provider == "groq":
        # По умолчанию для резюме берём ту же Llama-4-Scout, что и для SQL:
        # она даёт более качественные структурированные ответы с правильной
        # интерпретацией строк/колонок. TPM-бакет общий (30K), но при
        # max_tokens=1500 на резюме это укладывается в ~3 запроса/мин.
        # Если резюме упрётся в 429 — можно переключить GROQ_MODEL_SUMMARY
        # на openai/gpt-oss-20b (отдельный 8K бакет, 1000 t/s).
        summary_model = (
            os.getenv("GROQ_MODEL_SUMMARY")
            or "meta-llama/llama-4-scout-17b-16e-instruct"
        )
        summary_max_tokens = 1500
    elif provider == "anthropic":
        summary_model = (
            os.getenv("ANTHROPIC_MODEL_SUMMARY")
            or os.getenv("ANTHROPIC_MODEL")
            or "claude-haiku-4-5"
        )
        summary_max_tokens = 1500
    elif provider == "openai":
        summary_model = (
            os.getenv("OPENAI_MODEL_SUMMARY")
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        summary_max_tokens = 1500
    else:
        # Ollama — оставляем модель, выбранную пользователем в UI
        summary_model = model
        summary_max_tokens = 4096

    return sq.call_ollama(
        prompt,
        model=summary_model,
        url=ollama_url,
        max_tokens=summary_max_tokens,
    )


# ============================================================
# RAG PATH (использует answer_question.py)
# ============================================================

def run_rag(question: str, model_arg: str, top_k: int | None = None,
            candidates: int | None = None) -> dict:
    import answer_question as aq

    # Берём дефолты из .env (TOP_K / VECTOR_CANDIDATES), а не хардкод 5/30.
    if top_k is None:
        top_k = aq.DEFAULT_TOP_K
    if candidates is None:
        candidates = aq.DEFAULT_CANDIDATES

    selected_model = aq.resolve_ollama_model(model_arg)

    conn = aq.get_connection()
    try:
        t0 = time.time()
        chunks = aq.search_relevant_chunks(
            conn=conn,
            question=question,
            top_k=top_k,
            candidates=candidates,
            debug=False,
            use_rerank=True,
        )
        t_search = time.time() - t0

        if not chunks:
            return {
                "ok": True,
                "answer": "_Релевантных фрагментов в документах не найдено._",
                "chunks": [],
                "model": selected_model,
                "time_search": t_search,
                "time_llm": 0.0,
            }

        t0 = time.time()
        if aq.USE_OLLAMA_CHAT:
            messages = aq.build_chat_messages(question, chunks, model=selected_model)
            answer = aq.ask_ollama_chat(messages, model=selected_model, timeout=600)
        else:
            prompt = aq.build_prompt(question, chunks, model=selected_model)
            answer = aq.ask_ollama(prompt, model=selected_model, timeout=600)
        t_llm = time.time() - t0

        return {
            "ok": True,
            "answer": answer,
            "chunks": chunks,
            "model": selected_model,
            "time_search": t_search,
            "time_llm": t_llm,
        }
    finally:
        conn.close()


# ============================================================
# OLLAMA MODELS DISCOVERY
# ============================================================

# Хардкод-фолбэк на случай, если Ollama не отвечает или /api/tags
# отдал что-то странное. Это набор моделей, которые точно есть у тебя
# по тексту overview (qwen3:14b/8b, qwen3-vl:8b, glm4:9b, gemma3:12b,
# qwen2.5:7b). nomic-embed-text — embedding-модель, для LLM-запросов
# не нужна, исключаем.
_FALLBACK_MODELS = [
    "qwen3:14b",
    "qwen3:8b",
    "qwen2.5:7b",
    "glm4:9b",
    "gemma3:12b",
    "qwen3-vl:8b",
]


@st.cache_data(ttl=300, show_spinner=False)
def _list_ollama_models(ollama_url: str) -> list[str]:
    """
    Достаёт список моделей из Ollama /api/tags.

    Кэшируется на 5 минут, чтобы UI не дёргал Ollama на каждый клик.
    Если Ollama недоступна — возвращает _FALLBACK_MODELS.
    Embedding-модели (nomic-embed-text, bge-m3 и т.п.) отфильтровываем
    — они для LLM-вызовов не используются.
    """
    import requests

    tags_url = ollama_url.replace("/api/generate", "/api/tags")

    try:
        session = requests.Session()
        session.trust_env = False
        resp = session.get(tags_url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        names = [m.get("name") for m in (data.get("models") or []) if m.get("name")]
    except Exception:
        return _FALLBACK_MODELS

    # Фильтруем embedding-модели по имени.
    def is_llm(name: str) -> bool:
        lower = name.lower()
        return not any(
            tag in lower
            for tag in ("embed", "bge-m3", "nomic", "snowflake-arctic-embed")
        )

    llm_models = [n for n in names if is_llm(n)]
    return sorted(llm_models) if llm_models else _FALLBACK_MODELS


def _selectbox_with_default(label: str, options: list[str], default: str,
                            help: str | None = None, key: str | None = None) -> str:
    """
    Streamlit selectbox с попыткой найти default в options.
    Если default отсутствует — добавляет его в начало списка, чтобы
    пользователь не «слетел» с сохранённой настройки.
    """
    if default and default not in options:
        options = [default] + options
    try:
        idx = options.index(default) if default else 0
    except ValueError:
        idx = 0
    return st.selectbox(label, options, index=idx, help=help, key=key)


# ============================================================
# SESSION_STATE CALLBACKS
# ============================================================
# Streamlit запрещает менять session_state.X после того, как виджет
# с key=X создан в этом run'е. Поэтому используем on_click callback'и:
# они выполняются ДО создания виджета на следующем rerun.

def _cb_clear_question_input() -> None:
    """Очистка поля вопроса (по кнопке 'Очистить поле')."""
    st.session_state.question_input = ""


def _cb_append_company(name: str) -> None:
    """Дописать имя компании в поле вопроса (клик по кнопке в сайдбаре)."""
    current = st.session_state.get("question_input", "") or ""
    sep = " " if current and not current.endswith(" ") else ""
    st.session_state.question_input = f"{current}{sep}{name}"


def _cb_clear_chat() -> None:
    """Очистка истории чата."""
    st.session_state.chat_messages = []


def _cb_set_question(text: str) -> None:
    """Подставить готовый вопрос в поле ввода (для кнопок-примеров)."""
    st.session_state.question_input = text


# ============================================================
# COMPANIES DIRECTORY (для сайдбара)
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def _get_indicator_glossary() -> list[dict]:
    """
    Подтягивает финансовые показатели из finance.indicator_glossary
    для отображения в шапке приложения (расшифровка показателей).

    Кэш на 5 минут. Если БД недоступна — возвращает пустой список.
    """
    import psycopg2
    try:
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT"),
            dbname=os.getenv("PG_DB"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT code, display_name, description, formula, "
                    "       norm_range, direction, notes "
                    "FROM finance.indicator_glossary ORDER BY code;"
                )
                rows = cur.fetchall()
                return [
                    {
                        "code": r[0],
                        "display": r[1],
                        "description": r[2],
                        "formula": r[3],
                        "norm": r[4],
                        "direction": r[5],
                        "notes": r[6],
                    }
                    for r in rows
                ]
        finally:
            conn.close()
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def _list_companies() -> list[str]:
    """
    Подтягивает справочник компаний из finance.company.

    Кэш на минуту — между загрузками новых xlsx список редко меняется.
    Если БД недоступна — возвращает пустой список, UI не падает.
    """
    import psycopg2
    try:
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT"),
            dbname=os.getenv("PG_DB"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT short_name FROM finance.company ORDER BY short_name;"
                )
                return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()
    except Exception:
        return []


def _list_companies_with_segments() -> list[dict]:
    """
    Подтягивает компании с привязкой к сегменту (`finance.company.segment`).

    Возвращает list[{"name": str, "segment": str | None}]. Если колонка
    segment ещё не появилась — segment=None для всех. Используется
    в шапке UI для группированного списка и в Конструкторе вопроса
    для фильтрации компаний по сегменту.
    """
    import psycopg2
    try:
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT"),
            dbname=os.getenv("PG_DB"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
        )
        try:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        "SELECT short_name, segment FROM finance.company "
                        "ORDER BY segment NULLS LAST, short_name;"
                    )
                    return [
                        {"name": r[0], "segment": r[1]}
                        for r in cur.fetchall()
                    ]
                except Exception:
                    # segment ещё нет — fallback на простой список.
                    conn.rollback()
                    cur.execute(
                        "SELECT short_name FROM finance.company "
                        "ORDER BY short_name;"
                    )
                    return [
                        {"name": r[0], "segment": None}
                        for r in cur.fetchall()
                    ]
        finally:
            conn.close()
    except Exception:
        return []


# Стоп-слова, которые не несут информации о компании. Не пытаемся их
# сматчить — иначе fuzzy-поиск даст случайные совпадения по «как»,
# «оцени», «состояние» и т.п.
_COMPANY_STOPWORDS = {
    "как", "ты", "оцени", "оценишь", "оцениваешь", "оценить", "финансовое",
    "финансовый", "финансовая", "состояние", "положение", "ситуацию",
    "что", "какая", "какие", "какой", "сколько", "покажи", "сравни",
    "посчитай", "какова", "ао", "оао", "пао", "ооо", "зао", "по",
    "за", "у", "и", "в", "на", "с", "из", "от", "до", "год", "году",
    "годы", "годов", "выручка", "выручки", "прибыль", "прибыли", "убыток",
    "баланс", "активы", "обязательства", "капитал", "ликвидность",
    "рентабельность", "маржа", "ratio", "margin", "turnover",
}


def _extract_company_tokens(question: str) -> list[str]:
    """
    Достаёт из вопроса слова-кандидаты на имя компании.

    Грубая эвристика: токенизируем по не-буквам, отсекаем стоп-слова,
    отсекаем короткие токены (< 4 символов) и числа. Возвращаем
    нижне-регистровые токены — fuzzy-сматчер всё равно case-insensitive.
    """
    import re
    if not question:
        return []
    tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", question)
    out: list[str] = []
    for t in tokens:
        low = t.lower()
        if low in _COMPANY_STOPWORDS:
            continue
        if len(low) < 4:
            continue
        out.append(low)
    return out


def _suggest_companies(question: str, limit: int = 3) -> list[str]:
    """
    Fuzzy-подсказка имени компании на основе вопроса.

    1. Вытаскиваем «значимые» токены из вопроса.
    2. Для каждого токена прогоняем difflib против списка компаний из БД.
    3. Берём лучшие N уникальных совпадений с ratio ≥ 0.55.

    Возвращаем имена в том виде, как они хранятся в finance.company.short_name.
    """
    import difflib
    tokens = _extract_company_tokens(question)
    if not tokens:
        return []
    companies = _list_companies()
    if not companies:
        return []

    # Нормализованные имена компаний (без подчёркиваний, ё→е, нижний регистр)
    # для более точного fuzzy-match'а.
    def _norm(s: str) -> str:
        return s.replace("_", " ").replace("ё", "е").lower().strip()

    companies_norm = [(c, _norm(c)) for c in companies]

    scored: dict[str, float] = {}
    for tok in tokens:
        tok_n = tok.replace("ё", "е")
        for orig, norm in companies_norm:
            # Прямое вхождение токена в имя — высший приоритет.
            if tok_n in norm:
                scored[orig] = max(scored.get(orig, 0.0), 1.0)
                continue
            ratio = difflib.SequenceMatcher(None, tok_n, norm).ratio()
            # Так же пробуем сматчить против каждого слова в имени компании.
            for word in norm.split():
                r = difflib.SequenceMatcher(None, tok_n, word).ratio()
                if r > ratio:
                    ratio = r
            if ratio >= 0.55:
                scored[orig] = max(scored.get(orig, 0.0), ratio)

    return [c for c, _ in sorted(scored.items(), key=lambda x: -x[1])[:limit]]


# ============================================================
# UI
# ============================================================

st.set_page_config(page_title="Финансовый аналитик", layout="wide")

# Минимальный CSS — только убираем дефолтное ограничение ширины main
# контейнера и фиксируем ширину сайдбара. Шрифты оставляем дефолтными
# Streamlit (14px) — на 1920x1200 при 100% zoom это нормальный размер.
st.markdown(
    """
    <style>
    /* Главный контент: использует всю ширину за сайдбаром */
    [data-testid="stMainBlockContainer"],
    .main .block-container {
        max-width: 100% !important;
        padding: 1rem 2rem !important;
    }
    /* Сайдбар фиксированной ширины — иначе русские названия настроек
       режутся на 3-4 строки */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        width: 320px !important;
        min-width: 320px !important;
    }
    /* DataFrame: горизонтальный скролл при переполнении */
    [data-testid="stDataFrame"] {
        overflow-x: auto;
    }
    /* Кнопки компаний в expander сайдбара — компактнее, выровнены влево.
       Селектор уточнён, чтобы не цеплять Спросить/Очистить в main. */
    [data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button {
        padding: 0.25rem 0.55rem !important;
        text-align: left !important;
        font-size: 13px !important;
        line-height: 1.3 !important;
        white-space: normal !important;
        height: auto !important;
        min-height: 1.8rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Финансовый аналитик")
st.caption(
    "Ассистент по финансовой отчётности РСБУ. Спрашивай про выручку, "
    "прибыль, балансы и коэффициенты — ответы строятся SQL-запросами "
    "к схеме `finance.*` (23 компании)."
)

# ============================================================
# HEADER: компании в БД + примеры вопросов + расшифровка показателей
# ============================================================

# Видимый блок со списком компаний — менеджеру сразу понятен охват БД.
# С появлением c.segment группируем компании по отрасли — так быстрее
# виден баланс БД и удобнее искать. Если миграция segment ещё не
# применена (старый dev), у всех будет «(сегмент не задан)».
_companies_main = _list_companies()
_companies_with_segments = _list_companies_with_segments()
if _companies_main:
    # Группируем для UI: {segment: [name, ...]}
    _by_seg: dict[str, list[str]] = {}
    for _c in _companies_with_segments:
        _seg = _c.get("segment") or "(сегмент не задан)"
        _by_seg.setdefault(_seg, []).append(_c["name"])
    # Сортировка сегментов: по убыванию count, NULL-сегмент в конец.
    _seg_order = sorted(
        _by_seg.keys(),
        key=lambda s: (s == "(сегмент не задан)", -len(_by_seg[s]), s),
    )

    with st.expander(
        f"🏢 Компании в БД ({len(_companies_main)}) "
        f"· {len(_by_seg)} сегментов",
        expanded=False,
    ):
        st.caption(
            "_Полный список юрлиц с финансовой отчётностью в схеме "
            "`finance.*`, сгруппированный по отраслевому сегменту. "
            "Для подстановки в вопрос — пользуйся "
            "**🛠️ Конструктором вопроса** ниже._"
        )
        _md_lines: list[str] = []
        for _seg in _seg_order:
            _md_lines.append(
                f"**{_seg}** ({len(_by_seg[_seg])}):"
            )
            for _n in _by_seg[_seg]:
                _md_lines.append(f"- {_n}")
            _md_lines.append("")
        st.markdown("\n".join(_md_lines))


# ============================================================
# КОНСТРУКТОР ВОПРОСА
# ============================================================
# Три селектора: что узнать / компания / год. Кнопка собирает текстовый
# вопрос и кладёт его в session_state.question_input — оттуда его
# подхватывает основное текстовое поле ниже на странице.
_QUESTION_TEMPLATES: list[dict] = [
    {
        "label": "Выручка за год",
        "tpl": "Какая выручка {company} за {year} год?",
        "needs_year": True,
    },
    {
        "label": "Чистая прибыль за год",
        "tpl": "Какая чистая прибыль {company} за {year} год?",
        "needs_year": True,
    },
    {
        "label": "Operating Margin за год",
        "tpl": "Посчитай Operating Margin {company} за {year}",
        "needs_year": True,
    },
    {
        "label": "Quick Ratio на конец года",
        "tpl": "Покажи Quick Ratio {company} на конец {year}",
        "needs_year": True,
    },
    {
        "label": "Asset Turnover за год",
        "tpl": "Посчитай Asset Turnover {company} за {year}",
        "needs_year": True,
    },
    {
        "label": "Структура баланса на конец года",
        "tpl": (
            "Покажи структуру баланса {company} на конец {year}: "
            "внеоборотные, оборотные, капитал, обязательства"
        ),
        "needs_year": True,
    },
    {
        "label": "Динамика выручки 2022–2025",
        "tpl": "Покажи динамику выручки {company} за 2022-2025",
        "needs_year": False,
    },
    {
        "label": "Динамика чистой прибыли 2022–2025",
        "tpl": "Покажи динамику чистой прибыли {company} за 2022-2025",
        "needs_year": False,
    },
    {
        "label": "Все 5 коэффициентов за год",
        "tpl": (
            "Посчитай Operating Margin, Asset Turnover, Debtors Share, "
            "Quick Ratio и SGA Ratio для {company} за {year}"
        ),
        "needs_year": True,
    },
    {
        "label": "Финансовое состояние (комплекс)",
        "tpl": (
            "Покажи финансовое состояние {company}: выручка, чистая прибыль, "
            "балансовая стоимость за 2022-2025"
        ),
        "needs_year": False,
    },
    # ----- Шаблоны ПО СЕГМЕНТУ (компания заменяется на сегмент) -----
    {
        "label": "[Сегмент] Топ-3 по выручке за год",
        "tpl": "Топ-3 компаний сегмента «{segment}» по выручке за {year}",
        "needs_year": True,
        "needs_segment": True,
        "skip_company": True,
    },
    {
        "label": "[Сегмент] Средний Operating Margin за год",
        "tpl": (
            "Сравни средний Operating Margin компаний сегмента «{segment}» "
            "за {year}"
        ),
        "needs_year": True,
        "needs_segment": True,
        "skip_company": True,
    },
    {
        "label": "[Сегмент] Сравнение сегментов по марже",
        "tpl": (
            "Сравни средний Operating Margin по всем сегментам за {year}"
        ),
        "needs_year": True,
        "needs_segment": False,
        "skip_company": True,
    },
    {
        "label": "[Сегмент] Динамика сегмента 2022–2025",
        "tpl": (
            "Покажи динамику средней выручки сегмента «{segment}» "
            "за 2022-2025"
        ),
        "needs_year": False,
        "needs_segment": True,
        "skip_company": True,
    },
]


def _cb_build_question(template: dict, company_name: str, year: str,
                       segment: str) -> None:
    """Колбэк кнопки конструктора: собирает вопрос и кладёт в поле ввода."""
    # Имя компании для вопроса — без подчёркиваний (human-readable).
    company_pretty = (company_name or "").replace("_", " ").strip()
    kwargs: dict = {}
    if "{company}" in template["tpl"]:
        kwargs["company"] = company_pretty
    if template.get("needs_year"):
        kwargs["year"] = year
    if template.get("needs_segment"):
        kwargs["segment"] = segment
    q = template["tpl"].format(**kwargs)
    st.session_state.question_input = q


with st.expander("🛠️ Конструктор вопроса", expanded=False):
    if not _companies_main:
        st.caption(
            "_Конструктор недоступен: не удалось подгрузить список компаний из БД._"
        )
    else:
        # Cписок уникальных сегментов из БД для нового селектора.
        _qb_segments: list[str] = sorted({
            c.get("segment") for c in _companies_with_segments
            if c.get("segment")
        })

        _qb_col_tpl, _qb_col_co, _qb_col_seg, _qb_col_yr = st.columns(
            [3, 2, 2, 1]
        )

        with _qb_col_tpl:
            _tpl_label = st.selectbox(
                "Что узнать",
                options=[t["label"] for t in _QUESTION_TEMPLATES],
                key="_qb_template",
            )
        _selected_tpl = next(
            t for t in _QUESTION_TEMPLATES if t["label"] == _tpl_label
        )
        _tpl_needs_segment = _selected_tpl.get("needs_segment", False)
        _tpl_skip_company = _selected_tpl.get("skip_company", False)

        with _qb_col_seg:
            # Для шаблонов, где сегмент ОБЯЗАТЕЛЕН, опцию «(все)» убираем
            # — иначе пользователь соберёт вопрос с пустыми кавычками
            # «сегмента "" » и LLM не справится с фильтром. Для остальных
            # «(все)» — нормальный дефолт.
            if _tpl_needs_segment:
                _seg_options = _qb_segments or ["(нет сегментов в БД)"]
                _seg_index = 0
            else:
                _seg_options = ["(все)"] + _qb_segments
                _seg_index = 0
            _qb_segment = st.selectbox(
                "Сегмент",
                options=_seg_options,
                index=_seg_index,
                key="_qb_segment",
                help=(
                    "Фильтрует список компаний справа. Для сегментных "
                    "шаблонов «(все)» недоступно — выбери конкретный сегмент."
                ),
            )
            _qb_segment_value = "" if _qb_segment == "(все)" else _qb_segment

        with _qb_col_co:
            # Если выбран сегмент — отфильтровываем компании ТОЛЬКО этого
            # сегмента; иначе показываем все.
            if _qb_segment_value:
                _co_filtered = [
                    c["name"] for c in _companies_with_segments
                    if c.get("segment") == _qb_segment_value
                ]
            else:
                _co_filtered = _companies_main
            _qb_company = st.selectbox(
                "Компания",
                options=_co_filtered or ["—"],
                key="_qb_company",
                disabled=_tpl_skip_company,
                help=(
                    "Не используется для шаблонов «[Сегмент] ...»."
                    if _tpl_skip_company else
                    "Фильтруется выбором сегмента слева. Если сегмент = (все) "
                    "— доступны все 23 компании."
                ),
            )

        with _qb_col_yr:
            _qb_year = st.selectbox(
                "Год",
                options=["2022", "2023", "2024", "2025"],
                index=2,  # 2024 по умолчанию
                key="_qb_year",
                disabled=not _selected_tpl["needs_year"],
                help=(
                    "Для шаблонов с фиксированным периодом "
                    "(динамика / комплекс) год не используется."
                ),
            )

        # Валидация перед сборкой: если сегмент обязателен, но не задан —
        # дисэйблим кнопку и показываем предупреждение.
        _build_disabled = False
        _build_warning = ""
        if _tpl_needs_segment and not _qb_segment_value:
            _build_disabled = True
            _build_warning = (
                "Выбери конкретный сегмент — этот шаблон не работает с «(все)»."
            )
        if not _tpl_skip_company and _qb_company == "—":
            _build_disabled = True
            _build_warning = "В выбранном сегменте нет компаний."

        if _build_warning:
            st.caption(f"⚠️ _{_build_warning}_")

        st.button(
            "🔨 Собрать вопрос и подставить в поле",
            type="primary",
            use_container_width=True,
            key="_qb_build",
            disabled=_build_disabled,
            on_click=_cb_build_question,
            args=(_selected_tpl, _qb_company, _qb_year, _qb_segment_value),
        )

with st.expander("📊 Финансовые показатели — что считаем и почему", expanded=False):
    _glossary = _get_indicator_glossary()
    if not _glossary:
        st.caption(
            "_Справочник недоступен (БД недоступна или таблица "
            "`finance.indicator_glossary` ещё не создана)._"
        )
    else:
        st.caption(
            f"_Подтянуто из `finance.indicator_glossary` ({len(_glossary)} показателей). "
            f"Все формулы — эталонные: именно так считает SQL-маршрут._"
        )
        _direction_map = {
            "higher_better": "↑ больше — лучше",
            "lower_better": "↓ меньше — лучше",
            "optimal": "оптимально в диапазоне нормы",
        }
        # Две колонки для компактности
        _gl_col1, _gl_col2 = st.columns(2)
        for _idx, _indicator in enumerate(_glossary):
            _target_col = _gl_col1 if _idx % 2 == 0 else _gl_col2
            with _target_col:
                _direction = _direction_map.get(
                    _indicator["direction"], _indicator["direction"]
                )
                _norm_part = (
                    f"**Норма: {_indicator['norm']}** · {_direction}"
                    if _indicator.get("norm")
                    else f"{_direction}"
                )
                with st.container(border=True):
                    st.markdown(
                        f"**{_indicator['display']}**  \n"
                        f"*{_indicator['description']}*"
                    )
                    st.code(_indicator['formula'], language="text")
                    st.markdown(_norm_part)
                    if _indicator.get("notes"):
                        st.caption(f"💬 {_indicator['notes']}")

# --- Sidebar ---
with st.sidebar:
    st.header("Настройки")
    # Маршрут зафиксирован на SQL. RAG-маршрут отключён до доработки
    # документного пути — соответствующий код в проекте сохранён, но
    # из UI пока недоступен.
    override = "SQL (финансы)"
    st.caption(
        "_Все запросы идут в SQL по схеме `finance.*`. "
        "Документный RAG-маршрут пока отключён._"
    )
    st.divider()

    # ============================================================
    # МОДЕЛИ — провайдер-aware
    # ============================================================
    # Определяем активного провайдера из env (LLM_PROVIDER). На Streamlit
    # Cloud — groq, локально — ollama. Под каждый провайдер показываем
    # СВОИ модели с их реальными лимитами/особенностями.
    _provider = (os.getenv("LLM_PROVIDER") or "ollama").lower().strip()

    # RAG-маршрут отключён в продакшене — модель не выбираем.
    model_rag = "default"

    if _provider == "groq":
        st.subheader("Модели Groq")
        st.caption(
            "_Cloud-провайдер. Free-tier лимиты TPM (tokens per minute) "
            "ограничены — модели подобраны под наш промпт._"
        )

        # Список production+preview моделей Groq, подходящих под чат.
        # TPM указан для free-tier.
        _groq_sql_options = [
            "meta-llama/llama-4-scout-17b-16e-instruct",  # 30K TPM ⭐
            "llama-3.3-70b-versatile",                    # 12K TPM
            "openai/gpt-oss-120b",                        #  8K TPM
            "openai/gpt-oss-20b",                         #  8K TPM
            "qwen/qwen3-32b",                             #  6K TPM
            "llama-3.1-8b-instant",                       #  6K TPM
        ]
        _groq_summary_options = [
            "meta-llama/llama-4-scout-17b-16e-instruct",  # 30K TPM ⭐ дефолт
            "openai/gpt-oss-20b",                         #  8K TPM, 1000 t/s
            "openai/gpt-oss-120b",                        #  8K TPM, флагман
            "qwen/qwen3-32b",                             #  6K TPM
            "llama-3.3-70b-versatile",                    # 12K TPM
            "llama-3.1-8b-instant",                       #  6K TPM (слабая)
        ]
        _sql_default = (
            os.getenv("GROQ_MODEL")
            or "meta-llama/llama-4-scout-17b-16e-instruct"
        )
        model_sql = _selectbox_with_default(
            "SQL-модель",
            options=_groq_sql_options,
            default=_sql_default,
            help=(
                "Llama-4-Scout — самый большой TPM-бакет на free-tier (30K), "
                "хватает на наш тяжёлый промпт со схемой и few-shot примерами."
            ),
            key="model_sql",
        )
        _summary_default = (
            os.getenv("GROQ_MODEL_SUMMARY")
            or "meta-llama/llama-4-scout-17b-16e-instruct"
        )
        model_summary = _selectbox_with_default(
            "Модель резюме (после таблицы)",
            options=_groq_summary_options,
            default=_summary_default,
            help=(
                "Llama-4-Scout даёт качественные структурированные резюме. "
                "Если упрёшься в TPM-лимит — переключись на gpt-oss-20b "
                "(отдельный 8K бакет, 1000 t/s)."
            ),
            key="model_summary",
        )
    elif _provider == "anthropic":
        st.subheader("Модели Anthropic")
        _anthropic_options = [
            "claude-haiku-4-5",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
        ]
        _sql_default = os.getenv("ANTHROPIC_MODEL") or "claude-haiku-4-5"
        model_sql = _selectbox_with_default(
            "SQL-модель",
            options=_anthropic_options,
            default=_sql_default,
            key="model_sql",
        )
        _summary_default = (
            os.getenv("ANTHROPIC_MODEL_SUMMARY")
            or os.getenv("ANTHROPIC_MODEL")
            or "claude-haiku-4-5"
        )
        model_summary = _selectbox_with_default(
            "Модель резюме",
            options=_anthropic_options,
            default=_summary_default,
            key="model_summary",
        )
    elif _provider == "openai":
        st.subheader("Модели OpenAI")
        _openai_options = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]
        _sql_default = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        model_sql = _selectbox_with_default(
            "SQL-модель",
            options=_openai_options,
            default=_sql_default,
            key="model_sql",
        )
        _summary_default = (
            os.getenv("OPENAI_MODEL_SUMMARY")
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        model_summary = _selectbox_with_default(
            "Модель резюме",
            options=_openai_options,
            default=_summary_default,
            key="model_summary",
        )
    else:
        # Ollama — подтягиваем реальный список с локального сервера.
        st.subheader("Модели Ollama")
        _ollama_url_for_list = os.getenv(
            "OLLAMA_URL", "http://localhost:11434/api/generate"
        )
        _available_models = _list_ollama_models(_ollama_url_for_list)
        _sql_default = (
            os.getenv("OLLAMA_MODEL_SQL")
            or os.getenv("OLLAMA_MODEL")
            or "qwen3:14b"
        )
        model_sql = _selectbox_with_default(
            "SQL-модель",
            options=_available_models,
            default=_sql_default,
            help="Точность важнее скорости — лучше qwen3:14b.",
            key="model_sql",
        )
        _summary_default = (
            os.getenv("OLLAMA_MODEL_SUMMARY")
            or os.getenv("OLLAMA_MODEL_SQL")
            or os.getenv("OLLAMA_MODEL")
            or "qwen3:14b"
        )
        model_summary = _selectbox_with_default(
            "Модель резюме (после таблицы)",
            options=_available_models,
            default=_summary_default,
            help="Более лёгкая модель (qwen2.5:7b) уменьшит латентность в 3-4 раза.",
            key="model_summary",
        )

    st.divider()
    st.subheader("Дополнительно")
    explain_sql = st.checkbox(
        "Краткое резюме под таблицей (доп. вызов LLM)",
        value=True,
    )
    show_debug = st.checkbox(
        "Показывать SQL и список источников",
        value=True,
    )
    # RAG отключён — top_k не используется, но переменная нужна для
    # совместимости с обработчиком (на случай возврата RAG позже).
    top_k = int(os.getenv("TOP_K", "6") or "6")

    st.divider()
    st.subheader("Чат")
    # История чата ОТКЛЮЧЕНА: каждый вопрос обрабатывается независимо.
    # Раньше подмешивали последние 3 пары вопрос/ответ в промпт LLM —
    # это раздувало токены и упиралось в TPM-лимиты Groq. Сейчас follow-up
    # вопросы («а за 2023?») не сработают — пользователь должен явно
    # перепечатать компанию и период. Это сознательный трейд-офф ради
    # стабильности и предсказуемости.
    use_chat_history = False
    st.caption(
        "_История чата отображается на экране, но **не подмешивается в "
        "промпт LLM**. Каждый вопрос обрабатывается с нуля — для follow-up "
        "явно укажи компанию и период._"
    )
    st.button(
        "Очистить чат",
        use_container_width=True,
        on_click=_cb_clear_chat,
    )
    if st.session_state.get("chat_messages"):
        st.caption(f"_На экране: {len(st.session_state.chat_messages)} сообщений._")

    # Блок «Компании в БД» убран из сайдбара (он есть в шапке основной
    # области). Если понадобится вернуть — раскомментируй раздел в
    # git history.

# ============================================================
# CHAT STATE
# ============================================================
# Сообщения хранятся в session_state. Структура:
#   user-сообщение:
#     {"role": "user", "content": "выручка АО ЭКЗ за 2024"}
#   assistant-SQL:
#     {"role": "assistant", "route": "SQL", "reason": "...",
#      "sql": "...", "columns": [...], "rows": [...],
#      "summary": "...", "model": "glm4:9b",
#      "time_gen": 8.5, "time_exec": 0.02}
#   assistant-RAG:
#     {"role": "assistant", "route": "RAG", "reason": "...",
#      "content": "...", "chunks": [...], "model": "qwen3:14b",
#      "time_search": 0.5, "time_llm": 45.1}
#   assistant-ошибка:
#     {"role": "assistant", "route": chosen, "error": "...", "reason": "..."}

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

MAX_HISTORY_TURNS = 3  # учитываем последние N пар в контексте, если включено


# ============================================================
# RENDERING HELPERS
# ============================================================

# ============================================================
# Единицы измерения + экспорт результата в Excel
# ============================================================

# Множитель пересчёта ОТНОСИТЕЛЬНО тыс. руб (как суммы хранятся в БД).
_UNIT_FACTORS = {"руб": 1000.0, "тыс. руб": 1.0, "млн руб": 0.001, "млрд руб": 0.000001}
# Суффикс для переименования денежных колонок (…_tys_rub → …_<unit>).
_UNIT_SUFFIX = {"руб": "_rub", "тыс. руб": "_tys_rub", "млн руб": "_mln_rub", "млрд руб": "_mlrd_rub"}


def _is_amount_col(name: str) -> bool:
    """Денежная ли колонка (сумма в тыс. руб). Проценты/коэффициенты/годы — нет."""
    import re as _re
    n = str(name).lower()
    if any(t in n for t in ("_pct", "ratio", "turnover", "margin", "share")):
        return False
    return n.endswith("_tys_rub") or bool(_re.match(r"^c_\d", n))


def _scale_amounts(df, unit: str):
    """Копия df: денежные колонки пересчитаны в unit и переименованы
    (…_tys_rub → …_<unit>). Год/проценты/коэффициенты не трогаем."""
    from decimal import Decimal
    factor = _UNIT_FACTORS.get(unit, 1.0)
    out = df.copy()
    rename = {}
    for col in out.columns:
        if not _is_amount_col(col):
            continue
        if factor != 1.0:
            out[col] = out[col].apply(
                lambda v: (float(v) * factor)
                if v is not None and isinstance(v, (int, float, Decimal)) else v
            )
        if str(col).lower().endswith("_tys_rub"):
            rename[col] = col[: -len("_tys_rub")] + _UNIT_SUFFIX.get(unit, "_tys_rub")
    return out.rename(columns=rename) if rename else out


def _df_to_excel_bytes(df) -> bytes:
    """Сериализует DataFrame в .xlsx (реальные числа, не строки)."""
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Результат")
    return buf.getvalue()


def _render_sql_assistant(msg: dict, idx: int = 0) -> None:
    """Рендер ассистент-сообщения в SQL-маршруте: SQL → таблица → резюме."""
    st.caption(
        f"_Маршрут: **SQL** · модель: `{msg.get('model', '?')}` · "
        f"генерация {msg.get('time_gen', 0):.1f}s · "
        f"исполнение {msg.get('time_exec', 0):.2f}s · "
        f"причина роутера: {msg.get('reason', '—')}_"
    )

    if msg.get("error"):
        st.error(f"Ошибка SQL-пути: {msg['error']}")
        return

    if msg.get("sql"):
        with st.expander("SQL", expanded=False):
            st.code(msg["sql"], language="sql")

    if msg.get("rows"):
        import math
        from decimal import Decimal

        df = pd.DataFrame(msg["rows"], columns=msg["columns"])

        # Выбор единицы измерения — только если в результате есть денежные колонки.
        has_amounts = any(_is_amount_col(c) for c in df.columns)
        if has_amounts:
            unit = st.selectbox(
                "Единицы денежных сумм",
                options=["тыс. руб", "руб", "млн руб", "млрд руб"],
                index=0,
                key=f"unit_{idx}",
            )
        else:
            unit = "тыс. руб"
        df = _scale_amounts(df, unit) if has_amounts else df

        def _is_numlike(v):
            return isinstance(v, (int, float, Decimal))

        def _fmt_value(v, year_like: bool):
            if v is None:
                return None
            if isinstance(v, float) and math.isnan(v):
                return None
            if not _is_numlike(v):
                return str(v)
            try:
                f = float(v)
            except (TypeError, ValueError):
                return str(v)
            if year_like:
                return f"{int(f)}"
            if f == int(f):
                return f"{int(f):,}".replace(",", " ")
            return f"{f:,.2f}".replace(",", " ").replace(".", ",")

        df_display = df.copy()
        column_config = {}
        for col in df.columns:
            col_low = str(col).lower()
            is_year_like = (
                "year" in col_low or "год" in col_low or "period" in col_low
            )
            sample = df[col].head(5).tolist()
            is_num = any(_is_numlike(v) for v in sample if v is not None)
            if is_num:
                df_display[col] = df[col].apply(
                    lambda v, _y=is_year_like: _fmt_value(v, _y)
                )
            column_config[col] = st.column_config.TextColumn(col, width="medium")

        _height = min(35 * (len(df_display) + 1) + 3, 600)
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=_height,
            column_config=column_config or None,
        )
        # Экспорт результата (с учётом выбранной единицы) в Excel.
        try:
            st.download_button(
                "⬇️ Скачать в Excel",
                data=_df_to_excel_bytes(df),
                file_name="результат.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_{idx}",
            )
        except Exception as _e:
            st.caption(f"_Экспорт в Excel недоступен: {_e}_")

        dedup_note = ""
        if msg.get("deduped"):
            dedup_note = " · _дубли строк убраны автоматически (SQL без period_date в JOIN)_"
        unit_note = f" · _суммы в {unit}_" if has_amounts else ""
        st.caption(f"_{len(msg['rows'])} строк_{dedup_note}{unit_note}")
    elif msg.get("columns"):
        # Пустой результат — частая причина: пользователь ввёл компанию,
        # которой нет в БД (или она там под другим написанием — ё vs е,
        # пробел vs подчёркивание, без юр.формы и т.п.). Помогаем найти
        # ближайшую: пытаемся вытащить «значимое слово» из вопроса
        # пользователя и fuzzy-сматчить против справочника finance.company.
        _user_q = (msg.get("question") or "").strip()
        _suggestions = _suggest_companies(_user_q, limit=3)

        if _suggestions:
            _bullets = "\n".join(f"- **{c}**" for c in _suggestions)
            st.info(
                "Пустой результат — Postgres вернул 0 строк.\n\n"
                "Возможно, в БД эта компания под другим написанием. "
                "Похожие имена из справочника:\n\n"
                f"{_bullets}\n\n"
                "_Полный список — в шапке «Компании в БД»._"
            )
        else:
            st.info(
                "Пустой результат — Postgres вернул 0 строк.\n\n"
                "_Возможно, такой компании или периода нет в БД. "
                "Полный список компаний — в сайдбаре._"
            )

    if msg.get("summary"):
        st.markdown("**Резюме:**")
        st.write(msg["summary"])


def _render_rag_assistant(msg: dict) -> None:
    """Рендер ассистент-сообщения в RAG-маршруте: ответ + источники."""
    st.caption(
        f"_Маршрут: **RAG** · модель: `{msg.get('model', '?')}` · "
        f"поиск {msg.get('time_search', 0):.2f}s · "
        f"LLM {msg.get('time_llm', 0):.1f}s · "
        f"причина роутера: {msg.get('reason', '—')}_"
    )

    if msg.get("error"):
        st.error(f"Ошибка RAG-пути: {msg['error']}")
        return

    st.write(msg.get("content") or "")

    chunks = msg.get("chunks") or []
    if chunks:
        with st.expander(f"Источники ({len(chunks)} фрагментов)"):
            for i, ch in enumerate(chunks, 1):
                st.markdown(
                    f"**{i}. {ch['file_name']}** · "
                    f"стр. {ch.get('page', '—')} · "
                    f"тип: `{ch.get('chunk_type', '?')}` · "
                    f"раздел: _{ch.get('section_title') or '—'}_ · "
                    f"similarity: `{ch.get('similarity', 0):.3f}`"
                )
                preview = (ch.get("text") or "").strip()
                if len(preview) > 600:
                    preview = preview[:600] + "..."
                st.text(preview)
                st.divider()


def _render_assistant_message(msg: dict, idx: int = 0) -> None:
    """Диспетчер: какой рендер использовать."""
    if msg.get("route") == "SQL":
        _render_sql_assistant(msg, idx)
    else:
        _render_rag_assistant(msg)


def _build_history_context(max_turns: int = MAX_HISTORY_TURNS) -> str:
    """
    Собирает компактный текст истории SQL-диалога для подмешивания
    в промпт. Учитываем только сообщения с route=SQL (поскольку
    RAG-маршрут в проде отключён).

    Используется только если включён toggle "Учитывать историю".
    """
    if not st.session_state.chat_messages:
        return ""

    # Берём только SQL-обмены (user + assistant с route=SQL).
    sql_msgs = []
    for m in st.session_state.chat_messages:
        if m["role"] == "user":
            sql_msgs.append(m)
        elif m.get("route") == "SQL":
            sql_msgs.append(m)
    if not sql_msgs:
        return ""

    tail = sql_msgs[-(max_turns * 2):]
    lines = []
    for m in tail:
        if m["role"] == "user":
            lines.append(f"Пользователь: {m['content']}")
        else:
            short = m.get("summary") or "(SQL-результат без резюме)"
            short = short.strip().replace("\n", " ")
            if len(short) > 400:
                short = short[:400] + "..."
            lines.append(f"Ассистент: {short}")
    return "\n".join(lines)


def _extract_active_entities() -> dict:
    """
    Извлекает «активные сущности» из последних SQL-ответов: компания
    (или несколько), годы, показатели. Используем это как явный «scope»
    в промпте — LLM иначе игнорирует контекст и выдаёт общий SQL.

    Возвращает {"company": str | None, "years": list[str], "topic": str | None}.
    """
    msgs = st.session_state.get("chat_messages", []) or []
    if not msgs:
        return {"company": None, "years": [], "topic": None}

    company: str | None = None
    years: list[str] = []
    topic: str | None = None

    # Идём с конца, ищем последний SQL-ответ с rows и предыдущий user
    # вопрос рядом — берём из них short_name (если был) и упоминание года.
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("role") != "assistant":
            continue
        if m.get("route") != "SQL":
            continue
        if not m.get("rows"):
            continue
        # Найдём last short_name в rows (как самую вероятную «active company»).
        cols = m.get("columns") or []
        rows = m.get("rows") or []
        if "short_name" in cols:
            idx_company = cols.index("short_name")
            unique_companies = {
                str(r[idx_company]) for r in rows[:10] if r and r[idx_company]
            }
            if len(unique_companies) == 1:
                company = next(iter(unique_companies))
        # reporting_year тоже из cols.
        year_col = next(
            (c for c in cols if "year" in c.lower() or "год" in c.lower()),
            None,
        )
        if year_col:
            idx = cols.index(year_col)
            ys = sorted({str(int(r[idx])) for r in rows if r[idx] is not None})
            if ys:
                years = ys
        # Topic — из соседнего user-вопроса.
        if i > 0 and msgs[i - 1].get("role") == "user":
            topic = msgs[i - 1].get("content")
        break

    return {"company": company, "years": years, "topic": topic}


def _enrich_question_with_history(question: str) -> str:
    """
    Подмешивает историю + «активный scope» (компания/годы) в вопрос.

    Без явного scope LLM часто игнорирует контекст и пишет общий SQL.
    Чтобы это исправить, делаем отдельный блок «Активный контекст»,
    в котором перечисляем company / years из последнего SQL-ответа.
    """
    if not use_chat_history:
        return question
    history = _build_history_context()
    if not history:
        return question

    entities = _extract_active_entities()
    scope_lines: list[str] = []
    if entities["company"]:
        scope_lines.append(f"Активная компания: {entities['company']}")
    if entities["years"]:
        # Если периодов несколько — главный = последний (max).
        # Полный диапазон оставляем как hint, но фокус сразу на свежем годе.
        ys = entities["years"]
        if len(ys) == 1:
            scope_lines.append(f"Активный период: {ys[0]}")
        else:
            last = max(ys)
            other = ", ".join(y for y in ys if y != last)
            scope_lines.append(
                f"Активный период (фокус на ПОСЛЕДНЕМ): {last} "
                f"(также доступны: {other})"
            )
    if entities["topic"]:
        scope_lines.append(f"Предыдущий вопрос: «{entities['topic']}»")

    scope_block = ""
    if scope_lines:
        scope_block = (
            "АКТИВНЫЙ КОНТЕКСТ (если в новом вопросе явно не указана "
            "другая компания/период — используй ЭТИ значения как фильтры):\n"
            + "\n".join(f"- {x}" for x in scope_lines)
            + "\n\n"
        )

    return (
        f"Предыдущий диалог (для контекста):\n{history}\n\n"
        f"{scope_block}"
        f"Новый вопрос: {question}"
    )


# ============================================================
# RENDER CHAT HISTORY
# ============================================================

if st.session_state.chat_messages:
    for _i, msg in enumerate(st.session_state.chat_messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                _render_assistant_message(msg, _i)
    # Разделитель только когда уже есть история сообщений — иначе на
    # пустом старте чата висит лишняя горизонтальная черта.
    st.divider()

# ============================================================
# INPUT
# ============================================================

# Отложенная очистка поля вопроса (флаг ставит обработчик после
# успешной обработки нового вопроса). Это безопасно: происходит
# ДО создания виджета с key='question_input'.
if st.session_state.pop("_clear_input_pending", False):
    st.session_state.question_input = ""

question = st.text_area(
    "Вопрос:",
    height=180,
    placeholder=(
        "Примеры:\n"
        " — Какая выручка АО СИБКАБЕЛЬ за 2024 год?\n"
        " — Топ-5 компаний по балансу на конец 2024\n"
        " — Покажи динамику чистой прибыли ПАО НЛМК за 2022-2025\n"
        " — Посчитай Operating Margin АО ЭКЗ за 2024\n"
        "\n"
        "Можно собрать вопрос автоматически через 🛠️ Конструктор вопроса выше."
    ),
    key="question_input",
)

_col_go, _col_clear, _col_pad = st.columns([2, 2, 10])
with _col_go:
    go = st.button("Спросить", type="primary", use_container_width=True)
with _col_clear:
    st.button(
        "Очистить поле",
        use_container_width=True,
        on_click=_cb_clear_question_input,
    )

if go and question.strip():
    chosen, reason = route(question, override=override)

    # Сохраняем user-сообщение в истории сразу. Если ниже что-то упадёт,
    # пользователь хотя бы увидит свой вопрос в чате.
    st.session_state.chat_messages.append(
        {"role": "user", "content": question.strip()}
    )

    # Подмешиваем историю в вопрос, если toggle включён.
    effective_question = _enrich_question_with_history(question.strip())

    # Stash оригинальный вопрос пользователя в assistant_msg — нужен в
    # рендере для fuzzy-подсказок имён компаний при пустом результате.
    assistant_msg: dict = {
        "role": "assistant",
        "route": chosen,
        "reason": reason,
        "question": question.strip(),
    }
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

    try:
        # В продакшен-режиме все запросы идут в SQL. RAG-ветка
        # сохранена в коде (run_rag, _render_rag_assistant), но из UI
        # пока не доступна — до доработки документного пути.
        with st.spinner(f"`{model_sql}` пишет SQL, Postgres его исполняет..."):
            res = run_sql(
                effective_question, model=model_sql, ollama_url=ollama_url
            )

        assistant_msg.update({
            "sql": res.get("sql"),
            "columns": res.get("columns") or [],
            "rows": res.get("rows") or [],
            "model": model_sql,
            "time_gen": res.get("time_generate", 0.0),
            "time_exec": res.get("time_execute", 0.0),
            "deduped": res.get("deduped", False),
        })

        if explain_sql and res.get("rows"):
            try:
                with st.spinner(f"`{model_summary}` резюмирует данные..."):
                    summary = run_sql_summary(
                        effective_question,
                        res["columns"],
                        res["rows"],
                        model=model_summary,
                        ollama_url=ollama_url,
                    )
                assistant_msg["summary"] = summary
            except Exception as summary_exc:
                # Показываем полный текст исключения (для HTTPError от requests
                # сюда попадает status-код + URL + при необходимости тело
                # ответа Groq, которое подкладывает _call_groq). Это резко
                # упрощает диагностику в проде, где логи не у всех под рукой.
                exc_text = str(summary_exc) or repr(summary_exc)
                assistant_msg["summary"] = (
                    f"_Резюме не сгенерировалось: "
                    f"`{type(summary_exc).__name__}` — {exc_text}. "
                    f"Таблица выше — корректный результат запроса._"
                )

    except ValueError as exc:
        assistant_msg["error"] = f"Validation: {exc}"
    except Exception as exc:
        assistant_msg["error"] = f"{type(exc).__name__}: {exc}"

    st.session_state.chat_messages.append(assistant_msg)
    # Отложенная очистка поля: ставим флаг, на следующем rerun он
    # обнулит question_input ДО создания виджета.
    st.session_state._clear_input_pending = True
    st.rerun()
