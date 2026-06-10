"""
make_manifest_from_raw.py

Генератор CSV-манифестов для ingestion-пайплайна.

Зачем нужен:
Документы лежат в data/raw, разложенные по тематическим подпапкам
(например, "financial pdf (rus)", "market reports (ATH)"). Этот скрипт
обходит подпапки и собирает манифесты, которые понимает
run_ingestion_manifest.py — больше не нужно вести их вручную.

Логика полностью универсальна, без зашитых названий тем:
- каждая подпапка первого уровня внутри raw-root трактуется как отдельная
  тема (topic / source_group);
- файлы, лежащие прямо в raw-root без подпапки, попадают в группу "general".

Колонки манифеста:
    file_path,project,market,product,language,source_group,source_type,topic

Первые пять обязательны для run_ingestion_manifest.py, последние три —
опциональные: они пробрасывают тему документа в метаданные чанков, что
раньше терялось (в старых манифестах этих колонок не было).

Примеры:
    # манифест на каждую тему + общий манифест
    python make_manifest_from_raw.py

    # только общий манифест, и PDF, и Excel
    python make_manifest_from_raw.py --mode aggregate --extensions pdf,xlsx

    # перезаписать уже существующие манифесты
    python make_manifest_from_raw.py --force
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Порядок колонок в манифесте.
# Первые 5 — обязательные для run_ingestion_manifest.py,
# последние 3 — опциональные (тема документа -> метаданные чанков).
MANIFEST_COLUMNS = [
    "file_path",
    "project",
    "market",
    "product",
    "language",
    "source_group",
    "source_type",
    "topic",
]

GENERAL_GROUP = "general"

# Транслитерация кириллицы для slug — чтобы имена манифестов были ASCII.
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


# ============================================================
# HELPERS
# ============================================================

def slugify(value: str) -> str:
    """
    Делает безопасный ASCII-slug из произвольного имени папки.

    "market reports (rus, cable)" -> "market_reports_rus_cable"
    """

    value = value.strip().lower()

    out: list[str] = []
    for ch in value:
        if ch in _TRANSLIT:
            out.append(_TRANSLIT[ch])
        elif ch.isascii() and ch.isalnum():
            out.append(ch)
        else:
            out.append("_")

    slug = "".join(out)

    while "__" in slug:
        slug = slug.replace("__", "_")

    return slug.strip("_") or "group"


def normalize_extensions(value: str) -> set[str]:
    """Парсит '--extensions pdf,xlsx' в множество {'pdf', 'xlsx'}."""

    exts: set[str] = set()
    for raw in value.split(","):
        raw = raw.strip().lower().lstrip(".")
        if raw:
            exts.add(raw)
    return exts


def relative_file_path(path: Path) -> str:
    """
    file_path для манифеста: относительный POSIX-путь от корня проекта.

    run_ingestion_manifest.py резолвит относительные пути от PROJECT_ROOT,
    поэтому пишем именно относительный путь. Если файл вне проекта —
    оставляем абсолютный путь.
    """

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def make_row(
    file_path: Path,
    group_name: str,
    is_general: bool,
) -> dict[str, str]:
    """
    Строит одну строку манифеста для файла.

    source_group / topic берём как slug имени темы — это машинно-удобное
    значение, по которому потом можно фильтровать (embed_chunks.py
    --source-group, answer_question.py). market оставляем человекочитаемым.
    """

    slug = GENERAL_GROUP if is_general else slugify(group_name)

    return {
        "file_path": relative_file_path(file_path),
        # project / language заполняются в main() из аргументов.
        "project": "",
        "market": "" if is_general else group_name,
        "product": file_path.stem,
        "language": "",
        "source_group": slug,
        "source_type": file_path.suffix.lower().lstrip("."),
        "topic": slug,
    }


def collect_documents(
    raw_root: Path,
    extensions: set[str],
) -> list[dict[str, str]]:
    """
    Обходит raw_root и собирает строки манифеста.

    Подпапка первого уровня = отдельная тема. Файлы прямо в raw_root —
    группа "general". Внутри тематической подпапки файлы ищутся рекурсивно.
    """

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")
    if not raw_root.is_dir():
        raise NotADirectoryError(f"Raw root is not a directory: {raw_root}")

    def has_ok_ext(p: Path) -> bool:
        return p.is_file() and p.suffix.lower().lstrip(".") in extensions

    rows: list[dict[str, str]] = []

    # 1) Файлы прямо в raw_root (без тематической подпапки).
    loose_files = sorted(p for p in raw_root.iterdir() if has_ok_ext(p))
    for file_path in loose_files:
        rows.append(make_row(file_path, group_name=GENERAL_GROUP, is_general=True))

    # 2) Подпапки первого уровня = темы.
    subdirs = sorted(p for p in raw_root.iterdir() if p.is_dir())
    for subdir in subdirs:
        topic_files = sorted(p for p in subdir.rglob("*") if has_ok_ext(p))
        for file_path in topic_files:
            rows.append(make_row(file_path, group_name=subdir.name, is_general=False))

    return rows


def write_manifest(
    path: Path,
    rows: list[dict[str, str]],
    force: bool,
) -> bool:
    """
    Пишет один CSV-манифест. Возвращает True, если файл записан.

    Существующие файлы не перезаписываются без --force, чтобы случайно
    не затереть вручную подготовленный манифест.
    """

    if path.exists() and not force:
        print(f"  [SKIP]  {path.name} — уже существует (--force для перезаписи)")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)

    # utf-8-sig + comma — формат, который читает run_ingestion_manifest.py.
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in MANIFEST_COLUMNS})

    print(f"  [WRITE] {path.name} — {len(rows)} строк")
    return True


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate CSV ingestion manifests from a raw-documents folder tree. "
            "Each first-level subfolder is treated as a topic."
        )
    )

    parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Folder with raw documents (topic subfolders). Default: data/raw",
    )
    parser.add_argument(
        "--out-dir",
        default="data/manifests",
        help="Where to write manifest CSV files. Default: data/manifests",
    )
    parser.add_argument(
        "--mode",
        choices=["per-topic", "aggregate", "both"],
        default="both",
        help=(
            "per-topic: один манифест на каждую подпапку; "
            "aggregate: один общий манифест; both (по умолчанию): и то, и другое"
        ),
    )
    parser.add_argument(
        "--extensions",
        default="pdf",
        help="Comma-separated file extensions to include, e.g. 'pdf,xlsx'. Default: pdf",
    )
    parser.add_argument(
        "--project",
        default="RAG Analyst",
        help="Value for the 'project' column. Default: RAG Analyst",
    )
    parser.add_argument(
        "--language",
        default="ru",
        help=(
            "Value for the 'language' column for every row (ru/en/zh/mixed/unknown). "
            "Язык не определяется автоматически. Default: ru"
        ),
    )
    parser.add_argument(
        "--aggregate-name",
        default="all_pdf_context_manifest.csv",
        help="File name for the aggregate manifest. Default: all_pdf_context_manifest.csv",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite manifest files that already exist.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    raw_root = Path(args.raw_root)
    if not raw_root.is_absolute():
        raw_root = PROJECT_ROOT / raw_root

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    extensions = normalize_extensions(args.extensions)
    if not extensions:
        print("[ERROR] No valid extensions given via --extensions")
        sys.exit(1)

    print("=" * 80)
    print("MAKE MANIFEST FROM RAW")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw root:     {raw_root}")
    print(f"Out dir:      {out_dir}")
    print(f"Extensions:   {sorted(extensions)}")
    print(f"Mode:         {args.mode}")
    print(f"Project:      {args.project}")
    print(f"Language:     {args.language}")
    print(f"Force:        {args.force}")
    print("=" * 80)

    rows = collect_documents(raw_root, extensions)

    # Поля, одинаковые для всех строк.
    for row in rows:
        row["project"] = args.project
        row["language"] = args.language

    if not rows:
        print("No matching documents found. Nothing to write.")
        return

    # Группируем по source_group (slug темы).
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["source_group"], []).append(row)

    print(f"Documents found: {len(rows)}")
    for group, group_rows in sorted(groups.items()):
        print(f"  - {group}: {len(group_rows)}")
    print("=" * 80)

    written = 0
    skipped = 0

    if args.mode in ("per-topic", "both"):
        print("Per-topic manifests:")
        for group, group_rows in sorted(groups.items()):
            manifest_path = out_dir / f"{group}_manifest.csv"
            if write_manifest(manifest_path, group_rows, args.force):
                written += 1
            else:
                skipped += 1

    if args.mode in ("aggregate", "both"):
        print("Aggregate manifest:")
        aggregate_path = out_dir / args.aggregate_name
        if write_manifest(aggregate_path, rows, args.force):
            written += 1
        else:
            skipped += 1

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Manifests written: {written}")
    print(f"Manifests skipped: {skipped}")
    print("=" * 80)


if __name__ == "__main__":
    main()
