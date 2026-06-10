from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import psycopg2
import requests
from sentence_transformers import SentenceTransformer

# CrossEncoder импортируем лениво внутри _get_cross_encoder, чтобы запуск
# без USE_CROSS_ENCODER не тащил лишних 568 МБ модели в память.


# ============================================================
# PATHS / ENV
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"


def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        print(f"[WARN] .env file not found: {path}")
        return

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key:
                os.environ[key] = value


def get_env(*names: str, default=None):
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return default


load_env_file()


# ============================================================
# CONFIG
# ============================================================

POSTGRES_HOST = get_env("PG_HOST", "POSTGRES_HOST", "DB_HOST", "PGHOST", default="127.0.0.1")
POSTGRES_PORT = int(get_env("PG_PORT", "POSTGRES_PORT", "DB_PORT", "PGPORT", default="5433"))
POSTGRES_DB = get_env("PG_DB", "POSTGRES_DB", "DB_NAME", "PGDATABASE", default="rag_db")
POSTGRES_USER = get_env("PG_USER", "POSTGRES_USER", "DB_USER", "PGUSER", default="rag_user")
POSTGRES_PASSWORD = get_env("PG_PASSWORD", "POSTGRES_PASSWORD", "DB_PASSWORD", "PGPASSWORD", default="rag_pass")

EMBEDDING_MODEL_NAME = get_env("EMBEDDING_MODEL", "EMBEDDING_MODEL_NAME", default="BAAI/bge-m3")

# ВАЖНО: используем generate endpoint
OLLAMA_URL = get_env("OLLAMA_URL", default="http://localhost:11434/api/generate")

# Основная модель из .env
OLLAMA_MODEL = get_env("OLLAMA_MODEL", default="qwen3:14b")

# Алиасы моделей
OLLAMA_MODEL_FAST = get_env("OLLAMA_MODEL_FAST", default="qwen2.5:7b")
OLLAMA_MODEL_QUALITY = get_env("OLLAMA_MODEL_QUALITY", default=OLLAMA_MODEL)
OLLAMA_MODEL_CHINESE = get_env("OLLAMA_MODEL_CHINESE", default=OLLAMA_MODEL)
OLLAMA_MODEL_GLM = get_env("OLLAMA_MODEL_GLM", default="glm4:9b")
OLLAMA_MODEL_GEMMA = get_env("OLLAMA_MODEL_GEMMA", default="gemma3:12b")

DEFAULT_TOP_K = int(get_env("TOP_K", default="5"))
DEFAULT_CANDIDATES = int(get_env("VECTOR_CANDIDATES", default="30"))
DEFAULT_NUM_CTX = int(get_env("OLLAMA_NUM_CTX", default="8192"))

DEFAULT_PIPELINE = get_env("RAG_PIPELINE", default=None)
DEFAULT_PROJECT = get_env("RAG_PROJECT", default=None)
DEFAULT_MARKET = get_env("RAG_MARKET", default=None)
DEFAULT_PRODUCT = get_env("RAG_PRODUCT", default=None)
DEFAULT_SOURCE_FILE = get_env("RAG_SOURCE_FILE", default=None)


def resolve_ollama_model(model_arg: str | None) -> str:
    if not model_arg or model_arg == "default":
        return OLLAMA_MODEL

    aliases = {
        "fast": OLLAMA_MODEL_FAST,
        "quality": OLLAMA_MODEL_QUALITY,
        "chinese": OLLAMA_MODEL_CHINESE,
        "glm": OLLAMA_MODEL_GLM,
        "gemma": OLLAMA_MODEL_GEMMA,
    }

    return aliases.get(model_arg, model_arg)


def model_supports_no_think(model: str | None) -> bool:
    """
    Директива /no_think — управляющий токен семейства qwen3.

    Для qwen2.5 / glm4 / gemma3 он не распознаётся и попадает в prompt
    как обычный текст. Поэтому добавляем его только для qwen3.
    """
    return "qwen3" in (model or "").lower()


# ============================================================
# UTILS
# ============================================================

def timer(label: str):
    class _Timer:
        def __enter__(self):
            self.start = time.time()
            print(f"[START] {label}")
            return self

        def __exit__(self, exc_type, exc, tb):
            elapsed = time.time() - self.start
            if exc:
                print(f"[FAIL] {label}: {elapsed:.2f}s")
            else:
                print(f"[OK] {label}: {elapsed:.2f}s")

    return _Timer()


def print_config(selected_model: str | None = None) -> None:
    print("=" * 80)
    print("CONFIG")
    print("=" * 80)
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"ENV_PATH: {ENV_PATH}")
    print(f"POSTGRES_HOST: {POSTGRES_HOST}")
    print(f"POSTGRES_PORT: {POSTGRES_PORT}")
    print(f"POSTGRES_DB: {POSTGRES_DB}")
    print(f"POSTGRES_USER: {POSTGRES_USER}")
    print(f"POSTGRES_PASSWORD: {'*' * len(POSTGRES_PASSWORD) if POSTGRES_PASSWORD else None}")
    print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL_NAME}")
    print(f"OLLAMA_URL: {OLLAMA_URL}")
    print(f"OLLAMA_MODEL: {OLLAMA_MODEL}")
    print(f"OLLAMA_MODEL_FAST: {OLLAMA_MODEL_FAST}")
    print(f"OLLAMA_MODEL_QUALITY: {OLLAMA_MODEL_QUALITY}")
    print(f"OLLAMA_MODEL_CHINESE: {OLLAMA_MODEL_CHINESE}")
    print(f"OLLAMA_MODEL_GLM: {OLLAMA_MODEL_GLM}")
    print(f"OLLAMA_MODEL_GEMMA: {OLLAMA_MODEL_GEMMA}")
    if selected_model:
        print(f"SELECTED_OLLAMA_MODEL: {selected_model}")
    print(f"DEFAULT_TOP_K: {DEFAULT_TOP_K}")
    print(f"DEFAULT_CANDIDATES: {DEFAULT_CANDIDATES}")
    print(f"DEFAULT_PIPELINE: {DEFAULT_PIPELINE}")
    print(f"DEFAULT_PROJECT: {DEFAULT_PROJECT}")
    print(f"DEFAULT_MARKET: {DEFAULT_MARKET}")
    print(f"DEFAULT_PRODUCT: {DEFAULT_PRODUCT}")
    print(f"DEFAULT_SOURCE_FILE: {DEFAULT_SOURCE_FILE}")
    print("=" * 80)


def to_vector_literal(embedding) -> str:
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def safe_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def truncate_text(text: str, max_chars: int = 700) -> str:
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def normalize_text(value: str) -> str:
    value = value.lower()
    value = value.replace("ё", "е")
    value = value.replace("\u00a0", " ")
    value = re.sub(r"[^0-9a-zа-я\u4e00-\u9fff%]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def tokenize(value: str) -> list[str]:
    value = normalize_text(value)
    tokens = value.split()

    stopwords = {
        "и", "в", "во", "на", "по", "к", "ко", "с", "со", "за", "из", "от", "до",
        "для", "или", "а", "но", "что", "как", "какой", "какая", "какие", "какое",
        "это", "этот", "эта", "эти", "есть", "ли", "же", "у", "о", "об", "про",
        "рынок", "рынка", "рф", "россии", "году", "годы", "год",
    }

    return [token for token in tokens if len(token) >= 2 and token not in stopwords]


# ============================================================
# POSTGRES
# ============================================================

def get_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def test_postgres(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]

        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
            """
        )
        tables = [row[0] for row in cur.fetchall()]

        cur.execute(
            """
            SELECT 
                COUNT(*) AS total_chunks,
                COUNT(embedding) AS chunks_with_embeddings
            FROM chunks;
            """
        )
        total_chunks, chunks_with_embeddings = cur.fetchone()

        cur.execute(
            """
            SELECT
                COALESCE(c.metadata->>'pipeline', 'unknown') AS pipeline,
                COUNT(*) AS chunks_count,
                COUNT(c.embedding) AS embeddings_count
            FROM chunks c
            GROUP BY COALESCE(c.metadata->>'pipeline', 'unknown')
            ORDER BY chunks_count DESC;
            """
        )
        pipeline_stats = cur.fetchall()

    print(f"[DB] PostgreSQL: {version}")
    print(f"[DB] Tables: {tables}")
    print(f"[DB] Chunks: {total_chunks}")
    print(f"[DB] Chunks with embeddings: {chunks_with_embeddings}")

    print("[DB] Chunks by pipeline:")
    for pipeline, chunks_count, embeddings_count in pipeline_stats:
        print(f"  - {pipeline}: chunks={chunks_count}, embeddings={embeddings_count}")

    if "chunks" not in tables:
        raise RuntimeError("Table 'chunks' not found")

    if "documents" not in tables:
        raise RuntimeError("Table 'documents' not found")

    if total_chunks == 0:
        raise RuntimeError("No chunks in DB. Run load_rag_chunks_to_db.py first.")

    if chunks_with_embeddings == 0:
        raise RuntimeError("No embeddings in DB. Run embed_chunks.py first.")


# ============================================================
# OLLAMA
# ============================================================

def test_ollama(model_name: str | None = None) -> None:
    tags_url = OLLAMA_URL.replace("/api/generate", "/api/tags")

    session = requests.Session()
    session.trust_env = False

    response = session.get(tags_url, timeout=20)
    response.raise_for_status()

    data = response.json()
    models = data.get("models", [])

    print("[OLLAMA] Available models:")
    for model in models:
        print(f"  - {model.get('name')}")

    if model_name:
        model_names = {m.get("name") for m in models}

        if model_name not in model_names:
            print(f"[WARN] Model {model_name} not found in ollama list.")
            print(f"[WARN] Run: ollama pull {model_name}")


def ask_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 900) -> str:
    """
    /api/generate путь. Берёт плоский prompt от build_prompt().
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.05,
            "num_ctx": DEFAULT_NUM_CTX,
        },
    }

    session = requests.Session()
    session.trust_env = False

    response = session.post(
        OLLAMA_URL,
        json=payload,
        timeout=timeout,
    )

    response.raise_for_status()
    data = response.json()

    return data.get("response", "").strip()


def ask_ollama_chat(
    messages: list[dict],
    model: str = OLLAMA_MODEL,
    timeout: int = 900,
) -> str:
    """
    /api/chat путь — system+user сообщения раздельно. Это даёт qwen3
    более чёткое следование правилам, чем плоский prompt.

    Если у Ollama установлен старый /api/chat (возвращает 404 —
    исторически было так в этом проекте), автоматически откатываемся
    на /api/generate, склеив messages в один prompt.
    """

    chat_url = OLLAMA_URL.replace("/api/generate", "/api/chat")

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.05,
            "num_ctx": DEFAULT_NUM_CTX,
        },
    }

    session = requests.Session()
    session.trust_env = False

    try:
        response = session.post(chat_url, json=payload, timeout=timeout)

        if response.status_code == 404:
            raise RuntimeError("chat_404")

        response.raise_for_status()
        data = response.json()

        # /api/chat возвращает {"message": {"role": "assistant", "content": "..."}}
        msg = data.get("message") or {}
        return (msg.get("content") or "").strip()

    except (RuntimeError, requests.HTTPError) as exc:
        is_404 = (
            (isinstance(exc, RuntimeError) and str(exc) == "chat_404")
            or (
                isinstance(exc, requests.HTTPError)
                and exc.response is not None
                and exc.response.status_code == 404
            )
        )

        if not is_404:
            raise

        print("[OLLAMA] /api/chat unavailable (404), falling back to /api/generate")

        # Склеиваем system+user в один прямой prompt.
        joined = "\n\n".join(
            (m.get("content") or "")
            for m in messages
            if m.get("content")
        )
        return ask_ollama(joined, model=model, timeout=timeout)


USE_OLLAMA_CHAT = (get_env("OLLAMA_USE_CHAT", default="false") or "").lower() in (
    "1", "true", "yes", "on",
)


# ============================================================
# RERANK
# ============================================================

INTENT_PROFILES = {
    "import_export": {
        "positive": [
            "импорт", "экспорт", "вэд", "тн", "вед", "тамож", "страна", "страны",
            "происхожд", "назначен", "поставк", "ввоз", "вывоз", "usd",
            "янв", "нояб", "прирост", "код",
        ],
        "negative": [
            "прогноз", "2026", "2027", "2028", "инвестиции", "ввп",
        ],
        "chunk_type_boost": {
            "table": 0.04,
            "list": 0.04,
        },
    },
    "forecast": {
        "positive": [
            "прогноз", "2026", "2027", "2028", "плановый", "ожида", "динамик",
            "будущ", "сценар", "оценк",
        ],
        "negative": [
            "импорт", "экспорт", "тн", "вэд", "страна происхождения", "страна назначения",
        ],
        "chunk_type_boost": {
            "table": 0.05,
            "text": 0.02,
        },
    },
    "production": {
        "positive": [
            "производство", "объем", "объём", "выпуск", "категор", "номенклатур",
            "групп", "издел", "кабел", "провод", "динамик", "тонн", "км",
        ],
        "negative": [
            "импорт", "экспорт", "тн", "вэд", "страна происхождения", "страна назначения",
            "прогноз", "2026", "2027", "2028",
        ],
        "chunk_type_boost": {
            "table": 0.05,
        },
    },
    "finance": {
        "positive": [
            "выручка", "прибыль", "маржа", "актив", "обязательств", "долг", "капитал",
            "ликвидност", "рентабельност", "баланс", "опиу", "финанс",
        ],
        "negative": [],
        "chunk_type_boost": {
            "table": 0.05,
        },
    },
    "fire_safety": {
        "positive": [
            "пожар", "пожары", "погиб", "травм", "ущерб", "причин", "возгоран",
            "электрооборуд", "печ", "поджог", "неосторож", "огонь",
        ],
        "negative": [],
        "chunk_type_boost": {
            "table": 0.05,
            "text": 0.02,
        },
    },
    "ath": {
        "positive": [
            "ath", "гидроксид", "алюмини", "антипирен", "огнезащ", "阻燃剂",
            "氢氧化铝", "超细氢氧化铝", "低烟无卤", "lszh",
        ],
        "negative": [],
        "chunk_type_boost": {
            "table": 0.04,
            "text": 0.03,
        },
    },
}


def detect_intents(question: str) -> list[str]:
    q = normalize_text(question)
    intents = []

    for intent_name, profile in INTENT_PROFILES.items():
        hits = 0
        for keyword in profile["positive"]:
            if normalize_text(keyword) in q:
                hits += 1

        if hits > 0:
            intents.append(intent_name)

    return intents


def keyword_overlap_score(question: str, text: str) -> float:
    q_tokens = set(tokenize(question))
    if not q_tokens:
        return 0.0

    t = normalize_text(text)

    hits = 0
    for token in q_tokens:
        if token in t:
            hits += 1

    return hits / max(1, len(q_tokens))


def count_keyword_hits(keywords: list[str], text: str) -> int:
    text_norm = normalize_text(text)
    return sum(1 for keyword in keywords if normalize_text(keyword) in text_norm)


def _sigmoid(x: float) -> float:
    """sigmoid для нормализации raw logit-score cross-encoder'а в (0, 1)."""
    import math
    # Защита от overflow при больших отрицательных значениях.
    if x < -50:
        return 0.0
    if x > 50:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def rerank_chunks(question: str, chunks: list[dict], debug: bool = False) -> list[dict]:
    intents = detect_intents(question)

    reranked = []

    for ch in chunks:
        text = ch.get("text") or ""
        section = ch.get("section_title") or ""
        chunk_type = ch.get("chunk_type") or "unknown"
        similarity = float(ch.get("similarity") or 0.0)

        # База score'а: cross_score (через sigmoid → 0..1) если есть,
        # иначе исходный bge-m3 cosine similarity. Без этого keyword
        # overlap буст в 0.06-0.12 ничтожен против сырых logit'ов
        # реранкера (диапазон обычно -10..+10), и результат cross-encoder
        # фактически игнорировался бы.
        cross_score = ch.get("cross_score")
        if cross_score is not None:
            score = _sigmoid(float(cross_score))
        else:
            score = similarity

        body_overlap = keyword_overlap_score(question, text)
        section_overlap = keyword_overlap_score(question, section)

        score += body_overlap * 0.06
        score += section_overlap * 0.12

        intent_details = []

        for intent in intents:
            profile = INTENT_PROFILES[intent]

            positive_hits_section = count_keyword_hits(profile["positive"], section)
            positive_hits_body = count_keyword_hits(profile["positive"], text)

            negative_hits_section = count_keyword_hits(profile["negative"], section)
            negative_hits_body = count_keyword_hits(profile["negative"], text)

            intent_boost = 0.0
            intent_penalty = 0.0

            intent_boost += min(positive_hits_section, 4) * 0.045
            intent_boost += min(positive_hits_body, 6) * 0.015

            intent_penalty += min(negative_hits_section, 3) * 0.05
            intent_penalty += min(negative_hits_body, 4) * 0.015

            type_boost = profile.get("chunk_type_boost", {}).get(chunk_type, 0.0)

            score += intent_boost
            score += type_boost
            score -= intent_penalty

            intent_details.append({
                "intent": intent,
                "pos_section": positive_hits_section,
                "pos_body": positive_hits_body,
                "neg_section": negative_hits_section,
                "neg_body": negative_hits_body,
                "boost": round(intent_boost + type_boost, 4),
                "penalty": round(intent_penalty, 4),
            })

        question_norm = normalize_text(question)

        if chunk_type == "table" and any(x in question_norm for x in ["сколько", "какой", "какие", "динамик", "объем", "объём", "прогноз"]):
            score += 0.025

        if chunk_type == "list" and any(x in question_norm for x in ["почему", "причин", "страна", "страны", "экспорт", "импорт"]):
            score += 0.025

        ch2 = dict(ch)
        ch2["rerank_score"] = score
        ch2["rerank_debug"] = {
            "intents": intents,
            "body_overlap": round(body_overlap, 4),
            "section_overlap": round(section_overlap, 4),
            "intent_details": intent_details,
        }

        reranked.append(ch2)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    if debug:
        print("\n" + "=" * 80)
        print("RERANK DEBUG")
        print("=" * 80)
        print(f"Detected intents: {intents if intents else 'none'}")

        for i, ch in enumerate(reranked[:10], start=1):
            print(
                f"[{i}] vector={ch['similarity']:.4f} | "
                f"rerank={ch['rerank_score']:.4f} | "
                f"type={ch.get('chunk_type')} | "
                f"section={ch.get('section_title')}"
            )
            print(f"    debug={ch.get('rerank_debug')}")
            print("-" * 80)

    return reranked


# ============================================================
# DIVERSIFY (anti-cluster)
# ============================================================

DIVERSIFY_MAX_PER_GROUP = int(get_env("DIVERSIFY_MAX_PER_GROUP", default="2"))


def _chunk_group_key(ch: dict) -> tuple:
    """
    Возвращает «ключ группы» чанка для anti-cluster диверсификации.

    Идея: два чанка считаются «той же таблицей / тем же ближайшим
    контекстом», если у них совпадают source_file + section_title + либо
    table_title из metadata. Для table_part-серий (одна большая таблица,
    разрезанная chunker'ом на N окон) ключ получается одинаковым.

    Для текстовых чанков ключ опускаем до (source_file, section_title) —
    это разумная гранулярность: внутри одной секции одного документа
    обычно говорится про одно, и нет смысла отдавать LLM 5 чанков
    одного раздела.
    """

    source_file = ch.get("file_name") or ""
    section_title = ch.get("section_title") or ""
    chunk_type = ch.get("chunk_type") or ""

    metadata = ch.get("metadata") or {}
    source_metadata = ch.get("source_metadata") or {}

    table_title = (
        source_metadata.get("table_title")
        or metadata.get("table_title")
        or ""
    )

    return (source_file, section_title, chunk_type, table_title)


def diversify_chunks(
    chunks: list[dict],
    max_per_group: int = 2,
    debug: bool = False,
) -> list[dict]:
    """
    Anti-cluster диверсификация результата rerank.

    Идём по списку (уже отсортированному по rerank_score убыванию) и
    оставляем не более max_per_group чанков из одной «группы» (см.
    _chunk_group_key). Это спасает топ-K от ситуации «5 кусков одной
    финансовой формы вместо баланс+ОПиУ+пояснительной записки».

    Чанки, которые отбросили из-за лимита, не выкидываются — они
    собираются «в хвост» на случай, если top-K не наберётся другими
    группами. Так мы не делаем ответ хуже на запросах, где релевантна
    действительно одна таблица.

    max_per_group=2 — компромисс: достаточно, чтобы взять и актив и
    пассив одной формы, но не выдать в LLM пять окон одной финансовой
    таблицы.
    """

    if max_per_group <= 0 or not chunks:
        return chunks

    counts: dict[tuple, int] = {}
    primary: list[dict] = []
    overflow: list[dict] = []

    for ch in chunks:
        key = _chunk_group_key(ch)
        if counts.get(key, 0) < max_per_group:
            primary.append(ch)
            counts[key] = counts.get(key, 0) + 1
        else:
            overflow.append(ch)

    result = primary + overflow

    if debug:
        print("\n" + "=" * 80)
        print("DIVERSIFY DEBUG")
        print("=" * 80)
        print(f"max_per_group={max_per_group}, groups={len(counts)}, "
              f"primary={len(primary)}, overflow={len(overflow)}")
        for key, n in sorted(counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  {n}× {key}")

    return result


# ============================================================
# CROSS-ENCODER RERANK (bge-reranker-v2-m3)
# ============================================================
#
# Второй шаг retrieval поверх bge-m3 vector search. bge-m3 — bi-encoder:
# кодирует запрос и чанк независимо в один вектор по 1024 числа и сравнивает
# косинусом. Это быстро, но размывает специфический сигнал многословного
# запроса.
#
# bge-reranker-v2-m3 — cross-encoder: принимает пару (query, chunk)
# целиком и через self-attention видит, как каждое слово запроса
# относится к каждому слову чанка. Score намного точнее, особенно когда
# в чанке несколько похожих понятий или когда запрос длинный.
#
# Включается флагом .env: USE_CROSS_ENCODER=true
# По умолчанию выключен — модель ~568 МБ, не хочется грузить без нужды.

USE_CROSS_ENCODER = (get_env("USE_CROSS_ENCODER", default="false") or "").lower() in (
    "1", "true", "yes", "on",
)
CROSS_ENCODER_MODEL_NAME = get_env("CROSS_ENCODER_MODEL", default="BAAI/bge-reranker-v2-m3")
CROSS_ENCODER_MAX_LEN = int(get_env("CROSS_ENCODER_MAX_LEN", default="1024"))


# Кэш модели (lazy singleton). Грузится один раз при первом вызове
# cross_encoder_rerank(), потом переиспользуется при следующих запросах
# в том же процессе. Streamlit держит процесс между сообщениями —
# модель пройдёт по-настоящему только один раз за сессию UI.
_CROSS_ENCODER = None


def _get_cross_encoder():
    """
    Ленивая загрузка CrossEncoder. Если не удаётся загрузить —
    логируем и возвращаем None, чтобы вызвавший просто пропустил шаг.
    """
    global _CROSS_ENCODER

    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        print(f"[CROSS-ENCODER] sentence_transformers.CrossEncoder unavailable: {exc}")
        return None

    print(f"[CROSS-ENCODER] Loading model: {CROSS_ENCODER_MODEL_NAME}")
    print("[CROSS-ENCODER] (первая загрузка ~568 МБ + распаковка, дальше из кэша)")

    try:
        _CROSS_ENCODER = CrossEncoder(
            CROSS_ENCODER_MODEL_NAME,
            max_length=CROSS_ENCODER_MAX_LEN,
        )
    except Exception as exc:
        print(f"[CROSS-ENCODER] Load failed: {type(exc).__name__}: {exc}")
        return None

    return _CROSS_ENCODER


def cross_encoder_rerank(
    question: str,
    chunks: list[dict],
    debug: bool = False,
) -> list[dict]:
    """
    Прогоняет (query, chunk.text) через cross-encoder и сортирует
    чанки по полученному score'у.

    Поведение:
    - в каждый чанк добавляется поле "cross_score" (float или None).
    - вернётся новый список, отсортированный по cross_score убыванию;
      чанки без score (если модель не сработала) уходят в хвост в
      исходном порядке.
    - если модель недоступна (нет sentence_transformers / сеть упала
      на первой загрузке) — возвращаем исходный список без изменений.

    Не вызывает keyword/intent rerank — это отдельный шаг, который
    идёт ПОСЛЕ. Логика двух-этапного retrieval: cross-encoder делает
    основную пересортировку 50 кандидатов; rerank_chunks потом
    добавляет небольшие корректировки по intent/keyword.
    """

    if not chunks:
        return chunks

    encoder = _get_cross_encoder()
    if encoder is None:
        return chunks

    pairs = [(question, ch.get("text") or "") for ch in chunks]

    with timer(f"Cross-encoder rerank ({len(pairs)} pairs)"):
        try:
            scores = encoder.predict(pairs)
        except Exception as exc:
            print(f"[CROSS-ENCODER] predict failed: {type(exc).__name__}: {exc}")
            return chunks

    out: list[dict] = []
    for ch, score in zip(chunks, scores):
        ch2 = dict(ch)
        ch2["cross_score"] = float(score)
        out.append(ch2)

    out.sort(key=lambda x: x.get("cross_score") or float("-inf"), reverse=True)

    if debug:
        print("\n" + "=" * 80)
        print("CROSS-ENCODER RERANK DEBUG")
        print("=" * 80)
        for i, ch in enumerate(out[:10], start=1):
            print(
                f"[{i}] cross={ch['cross_score']:.4f} | "
                f"vec={ch['similarity']:.4f} | "
                f"type={ch.get('chunk_type')} | "
                f"section={ch.get('section_title')} | "
                f"file={ch['file_name']}"
            )

    return out


# ============================================================
# SEARCH
# ============================================================

def search_relevant_chunks(
    conn,
    question: str,
    top_k: int,
    candidates: int,
    debug: bool = False,
    pipeline: str | None = None,
    project: str | None = None,
    market: str | None = None,
    product: str | None = None,
    source_file: str | None = None,
    use_rerank: bool = True,
):
    with timer("Load embedding model"):
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    with timer("Create query embedding"):
        query_embedding = model.encode(
            question,
            normalize_embeddings=True,
        )

    query_vector = to_vector_literal(query_embedding)

    where_parts = ["c.embedding IS NOT NULL"]
    params: list[Any] = [query_vector]

    if pipeline:
        where_parts.append("c.metadata->>'pipeline' = %s")
        params.append(pipeline)

    if project:
        where_parts.append("c.metadata->>'project' = %s")
        params.append(project)

    if market:
        where_parts.append("c.metadata->>'market' = %s")
        params.append(market)

    if product:
        where_parts.append("c.metadata->>'product' = %s")
        params.append(product)

    if source_file:
        where_parts.append("d.file_name = %s")
        params.append(source_file)

    where_sql = " AND ".join(where_parts)

    sql = f"""
        SELECT
            d.file_name,
            c.document_id,
            c.page,
            c.chunk_index,
            c.chunk_text,
            c.metadata,
            1 - (c.embedding <=> %s::vector) AS similarity
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {where_sql}
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s;
    """

    params.append(query_vector)
    params.append(candidates)

    with timer("Vector search in Postgres"):
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

    chunks = []

    for row in rows:
        file_name, document_id, page, chunk_index, chunk_text, metadata, similarity = row

        metadata = safe_metadata(metadata)
        source_metadata = safe_metadata(metadata.get("source_metadata"))

        chunks.append(
            {
                "file_name": file_name,
                "document_id": document_id,
                "page": page,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "metadata": metadata,
                "source_metadata": source_metadata,
                "pipeline": metadata.get("pipeline"),
                "chunk_type": metadata.get("chunk_type"),
                "section_title": metadata.get("section_title"),
                "project": metadata.get("project"),
                "market": metadata.get("market"),
                "product": metadata.get("product"),
                "language": metadata.get("language"),
                "similarity": float(similarity) if similarity is not None else 0.0,
            }
        )

    # Шаг 1 (опционально) — cross-encoder rerank поверх bge-m3 кандидатов.
    # Делается ДО keyword/intent rerank, потому что cross-encoder даёт
    # самый точный «семантический» сигнал, а keyword/intent — это уже
    # эвристическая полировка поверх него (буст за table, intent profile
    # и т.п.).
    if USE_CROSS_ENCODER:
        chunks = cross_encoder_rerank(question=question, chunks=chunks, debug=debug)

    if use_rerank:
        chunks = rerank_chunks(question=question, chunks=chunks, debug=debug)

    # Диверсификация: не отдавать LLM >N кусков одной таблицы или одной
    # подряд идущей секции. Без неё топ-K часто оказывался разными
    # частями одной финансовой формы — а ОПиУ / контекстный текст
    # вообще не доезжали до prompt'а.
    chunks = diversify_chunks(chunks, max_per_group=DIVERSIFY_MAX_PER_GROUP, debug=debug)

    chunks = chunks[:top_k]

    if debug:
        print("\n" + "=" * 80)
        print("RETRIEVED CHUNKS")
        print("=" * 80)

        for i, ch in enumerate(chunks, start=1):
            preview = truncate_text(ch["text"], max_chars=900)
            rerank_score = ch.get("rerank_score")
            cross_score = ch.get("cross_score")

            score_part = (
                f" | rerank={rerank_score:.4f}"
                if rerank_score is not None
                else ""
            )
            cross_part = (
                f" | cross={cross_score:.4f}"
                if cross_score is not None
                else ""
            )

            print(
                f"[{i}] file={ch['file_name']} | "
                f"page={ch['page']} | "
                f"chunk={ch['chunk_index']} | "
                f"type={ch.get('chunk_type')} | "
                f"section={ch.get('section_title')} | "
                f"pipeline={ch.get('pipeline')} | "
                f"similarity={ch['similarity']:.4f}"
                f"{cross_part}"
                f"{score_part}"
            )
            print(preview)
            print("-" * 80)

    return chunks


# ============================================================
# PROMPT
# ============================================================

def _has_chinese(text: str) -> bool:
    """True, если в тексте есть китайские символы CJK."""
    if not text:
        return False
    for ch in text[:1000]:
        if "一" <= ch <= "鿿":
            return True
    return False


def _needs_cn_block(question: str, chunks: list[dict]) -> bool:
    """
    Решает, добавлять ли в промпт CN-словарь (ATH/万元/万吨).

    Подмешиваем CN-блок только когда он реально нужен:
    - вопрос задевает intent 'ath' (по INTENT_PROFILES), или
    - среди фрагментов есть chunk с китайскими символами, или
    - в chunk.language прописано 'zh' / 'mixed'.

    Это убирает шум промпта для вопросов про Сибкабель / ВНИИПО и т.п.,
    где правила про 万元 только мешают модели сосредоточиться.
    """

    if "ath" in detect_intents(question):
        return True

    for ch in chunks:
        lang = (ch.get("language") or "").lower()
        if lang in ("zh", "mixed"):
            return True
        if _has_chinese(ch.get("text") or ""):
            return True

    return False


_CN_GLOSSARY_BLOCK = """\
ЕСЛИ ВО ФРАГМЕНТАХ ЕСТЬ КИТАЙСКИЙ ТЕКСТ:
- Все названия организаций сохраняй на китайском + русский перевод. Не транслитерируй и не угадывай партнёров/университеты — пиши только то, что прямо в фрагменте.
- 万元 = 10 тысяч юаней (не «млн»). 万吨 = 10 тысяч тонн (не «млн тонн»). Сначала сохрани исходное значение «264.61 万吨», потом при необходимости пересчёт.
- ATH / 超细氢氧化铝 — это материал (ультратонкий гидроксид алюминия / антипирен), а не компания.
- 阻燃剂 = антипирен; 低烟无卤 = LSZH (низкодымный безгалогенный); 电线电缆 = провода и кабели; 覆铜板 = CCL (медно-фольгированный ламинат); 勃姆石 = бемит."""


_SYSTEM_PROMPT_BASE = """\
Ты аналитик. Извлекай факты СТРОГО из найденных фрагментов внутренней базы документов. Внешние знания, домыслы, чужие данные — запрещены.

Если ответа в фрагментах нет — пиши: «В найденных фрагментах нет данных для ответа». Не выдумывай причины, прогнозы, имена организаций, цифры и партнёров.

ИСТОЧНИКИ
- Ссылайся в формате [имя_файла, стр. X]. Если страница неизвестна — [имя_файла, стр. не определена].
- Никогда не пиши [Файл] / [Документ]. Бери имя из поля «Источник» фрагмента.
- Если фрагменты из разных проектов / рынков / продуктов — явно укажи это ограничение.
- Игнорируй явно нерелевантные фрагменты, опирайся на самые релевантные.

ЧИСЛА
- Указывай объект + период + единицу измерения.
- «тыс. руб.» оставляй как «тыс. руб.», не переименовывай в просто «руб.». Если в контексте «Суммы указаны в тысячах рублей» — все цифры из этой таблицы тоже тыс. руб.
- В млн / млрд переводи только когда уверен, и показывай оба значения: «3 573 265 тыс. руб. = 3,57 млрд руб.».
- Для сравнений проверяй направление: второе > первого → рост; второе < первого → снижение.

ТАБЛИЦЫ
- Если у фрагмента chunk_type=table, относись к нему как к таблице, не как к тексту.
- Не смешивай строки: каждая строка — отдельный показатель.
- Если просят общее состояние — дай вывод и 3–5 ключевых фактов, а не пересказ всех строк.
- Если table_quality низкое (weak / bad), или строки выглядят повторяющимися — укажи ограничение и не делай уверенный вывод.
- Для РСБУ-форм сохраняй коды строк (1600, 1200, 1300, 1500, 2110, 2200, 2400 и т.п.), если они есть в контексте.

ЯЗЫК И ФОРМАТ
- Всегда отвечай по-русски, даже если фрагменты на другом языке.
- Оригинальные иностранные термины — в скобках после русского перевода.
- Не выводи рассуждения, chain-of-thought или блоки <think>. Только итоговый ответ.
- Структура ответа: краткий вывод → ключевые факты с цитатами → ограничения (если есть)."""


def _build_system_prompt(question: str, chunks: list[dict]) -> str:
    """Системный промпт = базовая часть + (опционально) CN-блок."""
    parts = [_SYSTEM_PROMPT_BASE]
    if _needs_cn_block(question, chunks):
        parts.append("")
        parts.append(_CN_GLOSSARY_BLOCK)
    return "\n".join(parts)


def _build_context_block(chunks: list[dict]) -> str:
    """Контекст: компактные карточки фрагментов без лишних служебных полей."""
    blocks = []
    for i, ch in enumerate(chunks, start=1):
        metadata = safe_metadata(ch.get("metadata"))
        source_metadata = safe_metadata(ch.get("source_metadata"))

        chunk_type = ch.get("chunk_type") or metadata.get("chunk_type") or "unknown"
        section_title = ch.get("section_title") or metadata.get("section_title")
        table_quality = source_metadata.get("table_quality")
        page = ch["page"] if ch["page"] is not None else "не определена"

        # Только то, что реально нужно LLM: источник, страница, тип, раздел,
        # качество таблицы (если применимо), текст. Project/market/product/
        # pipeline убраны — это служебная для retrieval метадата, для ответа
        # она шум.
        header_lines = [
            f"[Фрагмент {i}]",
            f"Источник: {ch['file_name']}",
            f"Страница: {page}",
            f"Тип: {chunk_type}",
        ]
        if section_title:
            header_lines.append(f"Раздел: {section_title}")
        if chunk_type == "table" and table_quality:
            header_lines.append(f"Качество таблицы: {table_quality}")

        blocks.append("\n".join(header_lines) + "\n\nТекст:\n" + ch["text"])

    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, chunks: list[dict], model: str | None = None) -> str:
    """
    Собирает плоский prompt для /api/generate.

    Внутри:
    - системная часть (правила) генерится через _build_system_prompt;
    - CN-блок подмешивается только если он реально нужен (см. _needs_cn_block);
    - /no_think префикс — только для qwen3, остальные модели не понимают
      этот управляющий токен и пишут его как обычный текст.

    Если хочешь использовать /api/chat с раздельными system/user
    messages — есть build_chat_messages().
    """

    system = _build_system_prompt(question, chunks)
    context = _build_context_block(chunks)

    think_directive = "/no_think\n\n" if model_supports_no_think(model) else ""

    return (
        f"{think_directive}{system}\n\n"
        f"КОНТЕКСТ:\n{context}\n\n"
        f"ВОПРОС:\n{question}\n\n"
        f"ОТВЕТ:"
    )


def build_chat_messages(
    question: str,
    chunks: list[dict],
    model: str | None = None,
) -> list[dict]:
    """
    Сборка messages для /api/chat (system / user).

    Используется ask_ollama_chat. Преимущество — qwen3 (и большинство
    современных моделей) лучше следуют system role'у, чем правилам,
    зашитым в user-message. Если /api/chat недоступен, ask_ollama_chat
    откатится на /api/generate с тем же построенным prompt'ом.
    """

    system = _build_system_prompt(question, chunks)
    context = _build_context_block(chunks)

    # /no_think — управляющий токен qwen3. В chat-режиме его подмешиваем
    # в system, потому что в user-message он смотрится странно.
    if model_supports_no_think(model):
        system = "/no_think\n\n" + system

    user = f"КОНТЕКСТ:\n{context}\n\nВОПРОС:\n{question}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug RAG QA over Postgres/pgvector with Ollama"
    )

    parser.add_argument("question", type=str, help="Вопрос к базе документов")

    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Количество фрагментов для prompt")
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES, help="Количество кандидатов из vector search")
    parser.add_argument("--model", type=str, default="default", help="Модель Ollama или алиас: default, fast, quality, chinese, glm, gemma")

    parser.add_argument("--pipeline", type=str, default=DEFAULT_PIPELINE, help="Фильтр по pipeline")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help="Фильтр по project")
    parser.add_argument("--market", type=str, default=DEFAULT_MARKET, help="Фильтр по market")
    parser.add_argument("--product", type=str, default=DEFAULT_PRODUCT, help="Фильтр по product")
    parser.add_argument("--source-file", type=str, default=DEFAULT_SOURCE_FILE, help="Фильтр по documents.file_name")

    parser.add_argument("--debug", action="store_true", help="Показать диагностическую информацию")
    parser.add_argument("--no-llm", action="store_true", help="Только поиск, без генерации ответа")
    parser.add_argument("--skip-ollama-check", action="store_true", help="Не проверять Ollama API перед поиском")
    parser.add_argument("--no-rerank", action="store_true", help="Отключить keyword/metadata rerank")

    args = parser.parse_args()

    selected_model = resolve_ollama_model(args.model)

    question = args.question.strip()

    if not question:
        print("[ERROR] Empty question")
        sys.exit(1)

    use_rerank = not args.no_rerank

    print("=" * 80)
    print("RAG ANALYST DEBUG")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Top K: {args.top_k}")
    print(f"Candidates: {args.candidates}")
    print(f"Ollama model arg: {args.model}")
    print(f"Selected Ollama model: {selected_model}")
    print(f"Pipeline filter: {args.pipeline}")
    print(f"Project filter: {args.project}")
    print(f"Market filter: {args.market}")
    print(f"Product filter: {args.product}")
    print(f"Source file filter: {args.source_file}")
    print(f"Use rerank: {use_rerank}")
    print("=" * 80)

    if args.debug:
        print_config(selected_model=selected_model)

    if not args.no_llm and not args.skip_ollama_check:
        with timer("Check Ollama API"):
            test_ollama(selected_model)

    with timer("Connect to Postgres"):
        conn = get_connection()

    try:
        with timer("Check Postgres tables and embeddings"):
            test_postgres(conn)

        chunks = search_relevant_chunks(
            conn=conn,
            question=question,
            top_k=args.top_k,
            candidates=args.candidates,
            debug=args.debug,
            pipeline=args.pipeline,
            project=args.project,
            market=args.market,
            product=args.product,
            source_file=args.source_file,
            use_rerank=use_rerank,
        )

        if not chunks:
            print("[ERROR] No relevant chunks found")
            return

        if args.no_llm:
            print("[OK] Search completed. LLM generation skipped because --no-llm was used.")
            return

        with timer("Build prompt"):
            if USE_OLLAMA_CHAT:
                messages = build_chat_messages(question, chunks, model=selected_model)
                prompt_chars = sum(len(m.get("content") or "") for m in messages)
                print(f"[PROMPT] Chat messages: {len(messages)}, total chars: {prompt_chars}")
            else:
                prompt = build_prompt(question, chunks, model=selected_model)
                print(f"[PROMPT] Characters: {len(prompt)}")

        if args.debug:
            print("\n" + "=" * 80)
            print("PROMPT PREVIEW")
            print("=" * 80)
            if USE_OLLAMA_CHAT:
                for m in messages:
                    print(f"--- {m['role']} ---")
                    print((m.get('content') or '')[:2500])
                    print()
            else:
                print(prompt[:5000])
                print("\n...[prompt truncated in debug preview]...")

        with timer("Ask Ollama"):
            if USE_OLLAMA_CHAT:
                answer = ask_ollama_chat(messages, model=selected_model, timeout=900)
            else:
                answer = ask_ollama(prompt, model=selected_model, timeout=900)

        print("\n" + "=" * 80)
        print("ОТВЕТ")
        print("=" * 80)
        print(answer)

        print("\n" + "=" * 80)
        print("ИСПОЛЬЗОВАННЫЕ ФРАГМЕНТЫ")
        print("=" * 80)

        for i, ch in enumerate(chunks, start=1):
            rerank_score = ch.get("rerank_score")
            rerank_text = f" | rerank={rerank_score:.4f}" if rerank_score is not None else ""

            print(
                f"[{i}] {ch['file_name']} | "
                f"page={ch['page']} | "
                f"chunk={ch['chunk_index']} | "
                f"type={ch.get('chunk_type')} | "
                f"section={ch.get('section_title')} | "
                f"similarity={ch['similarity']:.4f}"
                f"{rerank_text}"
            )

    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR")
        print("=" * 80)
        print(type(e).__name__)
        print(e)
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()