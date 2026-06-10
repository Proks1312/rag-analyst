"""
Удалить документы из корпуса.

Чистит три места:
1. Postgres — documents + связанные chunks и extracted_tables.
2. data/processed/{ingestion,embedding_input,rag_chunks}/ — jsonl-артефакты.
3. data/manifests/*.csv — строки, ссылающиеся на эти файлы.

Идемпотентно. Имеет --dry-run, который показывает план без записи.

Использование:
    # Сухой прогон — посмотреть план:
    python remove_documents.py --files-from to_remove.txt --dry-run

    # Реально удалить:
    python remove_documents.py --files-from to_remove.txt

    # Через CLI-список:
    python remove_documents.py --files "АО_ЭКЗ-2026-04-16.pdf,ПАО_НЛМК-2026-04-16.pdf"

    # Удалить только из БД, оставить файлы и манифесты:
    python remove_documents.py --files-from to_remove.txt --keep-processed --keep-manifests

Перед запуском ОСТАНОВИ embed_chunks.py (Ctrl+C в его окне), иначе
DELETE'ы будут конкурировать с его UPDATE'ами за блокировки.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent

INGESTION_DIR = PROJECT_ROOT / "data" / "processed" / "ingestion"
EMBEDDING_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "embedding_input"
RAG_CHUNKS_DIR = PROJECT_ROOT / "data" / "processed" / "rag_chunks"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"


# ============================================================
# TARGETS
# ============================================================

def load_targets(args) -> list[str]:
    """Возвращает отсортированный список basename'ов (с расширением)."""
    targets: list[str] = []

    if args.files:
        targets.extend(s.strip() for s in args.files.split(",") if s.strip())

    if args.files_from:
        path = Path(args.files_from)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                targets.append(line)

    return sorted(set(targets))


# ============================================================
# POSTGRES
# ============================================================

def get_conn():
    import psycopg2

    load_dotenv(PROJECT_ROOT / ".env")
    return psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
    )


def delete_from_db(conn, file_name: str, dry_run: bool) -> tuple[int, int]:
    """
    Возвращает (documents_count, chunks_count). Может быть несколько
    documents с одним file_name (история перезаливок).
    """

    with conn.cursor() as cur:
        cur.execute("SELECT id FROM documents WHERE file_name = %s", (file_name,))
        ids = [row[0] for row in cur.fetchall()]
        if not ids:
            return 0, 0

        cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ANY(%s)",
            (ids,),
        )
        chunks_count = int(cur.fetchone()[0])

        if dry_run:
            return len(ids), chunks_count

        # extracted_tables может не существовать, обернём в try
        try:
            cur.execute(
                "DELETE FROM extracted_tables WHERE document_id = ANY(%s)",
                (ids,),
            )
        except Exception:
            conn.rollback()
        cur.execute("DELETE FROM chunks WHERE document_id = ANY(%s)", (ids,))
        cur.execute("DELETE FROM documents WHERE id = ANY(%s)", (ids,))

        return len(ids), chunks_count


# ============================================================
# PROCESSED FILES
# ============================================================

def remove_processed_files(stem: str, dry_run: bool) -> list[Path]:
    candidates = [
        INGESTION_DIR / f"{stem}.blocks.jsonl",
        INGESTION_DIR / f"{stem}.tables.jsonl",
        INGESTION_DIR / f"{stem}.visuals.jsonl",
        EMBEDDING_INPUT_DIR / f"{stem}.embedding_records.jsonl",
        RAG_CHUNKS_DIR / f"{stem}.rag_chunks.jsonl",
    ]
    removed: list[Path] = []
    for fp in candidates:
        if fp.exists():
            if not dry_run:
                fp.unlink()
            removed.append(fp)
    return removed


# ============================================================
# MANIFESTS
# ============================================================

def remove_from_manifests(
    file_names: set[str],
    dry_run: bool,
) -> dict[str, int]:
    """
    Удаляет строки из всех CSV-манифестов в data/manifests/, где
    basename file_path входит в file_names.
    """

    result: dict[str, int] = {}

    for manifest_path in sorted(MANIFESTS_DIR.glob("*.csv")):
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            continue

        header = rows[0]
        try:
            file_path_idx = header.index("file_path")
        except ValueError:
            continue

        kept: list[list[str]] = [header]
        removed_n = 0
        for row in rows[1:]:
            if file_path_idx >= len(row):
                kept.append(row)
                continue
            basename = re.split(r"[\\/]", row[file_path_idx])[-1].strip().strip('"')
            if basename in file_names:
                removed_n += 1
            else:
                kept.append(row)

        if removed_n > 0:
            if not dry_run:
                with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(kept)
            result[manifest_path.name] = removed_n

    return result


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove documents from corpus (DB + processed + manifests)."
    )
    parser.add_argument(
        "--files",
        help="Список файлов через запятую (basename с расширением).",
    )
    parser.add_argument(
        "--files-from",
        help="Путь к txt со списком (по одному на строку; # для комментариев).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать план, ничего не удалять.",
    )
    parser.add_argument(
        "--keep-manifests",
        action="store_true",
        help="Не трогать манифесты.",
    )
    parser.add_argument(
        "--keep-processed",
        action="store_true",
        help="Не трогать processed-jsonl (только БД).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = load_targets(args)

    if not targets:
        print("Список файлов пуст. Дай --files или --files-from.")
        return

    print("=" * 78)
    print("REMOVE DOCUMENTS")
    print("=" * 78)
    print(f"Файлов: {len(targets)}")
    for t in targets:
        print(f"  - {t}")
    print(f"Dry-run:       {args.dry_run}")
    print(f"Keep manifests: {args.keep_manifests}")
    print(f"Keep processed: {args.keep_processed}")
    print("=" * 78)

    # 1) Postgres
    print("\n[1/3] Postgres")
    conn = get_conn()
    try:
        total_docs = 0
        total_chunks = 0
        for fn in targets:
            n_docs, n_chunks = delete_from_db(conn, fn, args.dry_run)
            mark = "WOULD" if args.dry_run else " OK  "
            print(f"  {mark}  {fn:50s}  docs={n_docs:>2}  chunks={n_chunks:>5}")
            total_docs += n_docs
            total_chunks += n_chunks
        if not args.dry_run:
            conn.commit()
        print(f"  ИТОГО: documents={total_docs}, chunks={total_chunks}")
    finally:
        conn.close()

    # 2) Processed jsonl
    if args.keep_processed:
        print("\n[2/3] Processed jsonl — пропущено (--keep-processed)")
    else:
        print("\n[2/3] data/processed/* artifacts")
        total_files = 0
        for fn in targets:
            stem = Path(fn).stem
            removed = remove_processed_files(stem, args.dry_run)
            mark = "WOULD" if args.dry_run else " OK  "
            for fp in removed:
                try:
                    rel = fp.relative_to(PROJECT_ROOT)
                except ValueError:
                    rel = fp
                print(f"  {mark}  {rel}")
            total_files += len(removed)
        print(f"  ИТОГО файлов: {total_files}")

    # 3) Манифесты
    if args.keep_manifests:
        print("\n[3/3] Манифесты — пропущено (--keep-manifests)")
    else:
        print("\n[3/3] data/manifests/*.csv")
        result = remove_from_manifests(set(targets), args.dry_run)
        mark = "WOULD" if args.dry_run else " OK  "
        if not result:
            print("  (ничего не найдено в манифестах)")
        else:
            for mfn, n in sorted(result.items()):
                print(f"  {mark}  {mfn}: {n} строк")

    print()
    print("=" * 78)
    print("DONE" if not args.dry_run else "DRY-RUN COMPLETE — ничего не изменено")
    print("=" * 78)


if __name__ == "__main__":
    main()
