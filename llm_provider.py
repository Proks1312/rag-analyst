"""
Универсальный LLM-провайдер.

Локально использует Ollama, в облаке (Streamlit Cloud) — внешние API:
Groq / Anthropic / OpenAI. Переключение через переменную окружения
LLM_PROVIDER:

    LLM_PROVIDER=ollama     # default — для локальной разработки
    LLM_PROVIDER=groq       # для бесплатного публичного деплоя
    LLM_PROVIDER=anthropic  # для коммерческой точности (Claude)
    LLM_PROVIDER=openai     # альтернатива

Все провайдеры используют HTTPS + requests, никаких тяжёлых SDK — это
важно для Streamlit Cloud, где каждый пакет съедает диск и память.

Использование:
    from llm_provider import call_llm
    answer = call_llm(prompt="Напиши SQL для выручки", system="Ты ассистент")
"""

from __future__ import annotations

import os
from typing import Optional

import requests


DEFAULT_PROVIDER = "ollama"
DEFAULT_TIMEOUT = 900


def get_provider() -> str:
    """Текущий провайдер из env."""
    return (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower().strip()


_OLLAMA_ALIASES = {"default", "fast", "quality", "gemma", "glm", "chinese"}


def _is_ollama_style_model(model: Optional[str]) -> bool:
    """
    Является ли имя модели «Ollama-стиля»?

    Ollama использует имена с ':' (qwen3:14b, glm4:9b, llama2:7b), а UI
    Streamlit может ещё передавать алиасы вроде 'default' / 'fast'. Эти
    имена бессмысленны для облачных провайдеров (Groq, Anthropic, OpenAI),
    у них свои naming-conventions: llama-3.3-70b-versatile, claude-haiku-4-5,
    gpt-4o-mini и т.п.

    Эта функция помогает в call_llm() при провайдере != ollama сбросить
    переданную модель в None, чтобы дальше она взялась из env переменной
    (GROQ_MODEL / ANTHROPIC_MODEL / OPENAI_MODEL).
    """
    if not model:
        return False
    if ":" in model:
        return True
    if model.lower() in _OLLAMA_ALIASES:
        return True
    return False


def call_llm(
    prompt: str,
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """
    Универсальный вызов LLM.

    Аргументы:
        prompt — основной запрос
        model — имя модели (если None или Ollama-style на cloud — берётся из env)
        timeout — секунд на запрос
        system — опциональный system prompt (только chat-провайдеры)
        temperature — для определённости лучше 0.0
        max_tokens — лимит ответа (Groq/Anthropic/OpenAI)

    Возвращает строку с текстом ответа.
    """
    provider = get_provider()

    # Если провайдер cloud, а пришло Ollama-имя (qwen3:14b или default) —
    # сбрасываем в None, чтобы дальше взялась GROQ_MODEL / ANTHROPIC_MODEL /
    # OPENAI_MODEL из переменной окружения.
    if provider != "ollama" and _is_ollama_style_model(model):
        model = None

    if provider == "ollama":
        return _call_ollama(prompt, model, timeout, temperature)
    if provider == "groq":
        return _call_groq(prompt, model, timeout, system, temperature, max_tokens)
    if provider == "anthropic":
        return _call_anthropic(prompt, model, timeout, system, temperature, max_tokens)
    if provider == "openai":
        return _call_openai(prompt, model, timeout, system, temperature, max_tokens)

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. "
        "Допустимые: ollama, groq, anthropic, openai."
    )


# ============================================================
# OLLAMA
# ============================================================

def _call_ollama(prompt: str, model: Optional[str], timeout: int,
                 temperature: float) -> str:
    url = os.getenv("OLLAMA_URL") or "http://localhost:11434/api/generate"
    model = (
        model
        or os.getenv("OLLAMA_MODEL_SQL")
        or os.getenv("OLLAMA_MODEL")
        or "qwen3:14b"
    )
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX") or "16384")

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }
    session = requests.Session()
    session.trust_env = False
    response = session.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return (response.json().get("response") or "").strip()


# ============================================================
# GROQ (бесплатный для прода)
# ============================================================

def _call_groq(prompt: str, model: Optional[str], timeout: int,
               system: Optional[str], temperature: float,
               max_tokens: int) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY не задан. Получи ключ на https://console.groq.com/keys "
            "и пропиши в Streamlit Secrets или .env"
        )

    # Дефолт: llama-4-scout. У него на free-tier Groq самый большой
    # TPM-бакет среди приличных моделей (30K vs 12K у llama-3.3-70b
    # vs 8K у gpt-oss-120b). А SQL-промпт у нас тяжёлый (схема + glossary
    # + 10 few-shot примеров ~ 7-10K токенов на запрос), и на 12K-бакете
    # упираемся в 429 после 1-2 запросов в минуту. Llama-4 умнее Llama-3,
    # формально preview но Meta её стабилизировала.
    model = (
        model
        or os.getenv("GROQ_MODEL")
        or "meta-llama/llama-4-scout-17b-16e-instruct"
    )

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    # Если Groq вернул не-2xx, raise_for_status() кидает HTTPError, но
    # без тела ответа — а там как раз JSON с реальной причиной (например,
    # {"error":{"message":"model `qwen3:14b` does not exist",...}}).
    # Подкладываем тело в исключение, чтобы оно дошло и до UI, и до логов.
    if not response.ok:
        body = (response.text or "").strip()[:500]
        raise requests.HTTPError(
            f"Groq {response.status_code} for model={model!r}: {body}",
            response=response,
        )
    data = response.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


# ============================================================
# ANTHROPIC (Claude)
# ============================================================

def _call_anthropic(prompt: str, model: Optional[str], timeout: int,
                    system: Optional[str], temperature: float,
                    max_tokens: int) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY не задан. Получи ключ на "
            "https://console.anthropic.com/settings/keys"
        )

    model = model or os.getenv("ANTHROPIC_MODEL") or "claude-haiku-4-5"

    payload: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return (data["content"][0]["text"] or "").strip()


# ============================================================
# OPENAI
# ============================================================

def _call_openai(prompt: str, model: Optional[str], timeout: int,
                 system: Optional[str], temperature: float,
                 max_tokens: int) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY не задан. Получи ключ на "
            "https://platform.openai.com/api-keys"
        )

    model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return (data["choices"][0]["message"]["content"] or "").strip()
