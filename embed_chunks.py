from __future__ import annotations

import argparse
import os
from typing import Any

import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()


# Имя embedding-модели берём из .env, чтобы оно было ОДИНАКОВЫМ с answer_question.py.
# Это критично для RAG: чанки и запрос обязаны кодироваться одной моделью,
# иначе вектора лежат в разных пространствах и поиск молча ломается.
# Поддерживаем те же имена переменных и тот же fallback, что и answer_question.py.
DEFAULT_MODEL_NAME = (
    os.getenv("EMBEDDING_MODEL")
    or os.getenv("EMBEDDING_MODEL_NAME")
    or "BAAI/bge-m3"
)
DEFAULT_BATCH_SIZE = 16
DEFAULT_FETCH_PAGE_SIZE = 1000


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable is not set: {name}")
    return value


def get_conn():
    return psycopg2.connect(
        host=get_required_env("PG_HOST"),
        port=get_required_env("PG_PORT"),
        dbname=get_required_env("PG_DB"),
        user=get_required_env("PG_USER"),
        password=get_required_env("PG_PASSWORD"),
    )


def count_chunks_without_embeddings(
    conn,
    source_group: str | None = None,
) -> int:
    with conn.cursor() as cur:
        if source_group:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                WHERE embedding IS NULL
                  AND metadata->>'source_group' = %s;
                """,
                (source_group,),
            )
        else:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                WHERE embedding IS NULL;
                """
            )

        return int(cur.fetchone()[0])


def load_chunks_without_embeddings(
    conn,
    limit: int,
    source_group: str | None = None,
) -> list[tuple[int, str]]:
    with conn.cursor() as cur:
        if source_group:
            cur.execute(
                """
                SELECT id, chunk_text
                FROM chunks
                WHERE embedding IS NULL
                  AND metadata->>'source_group' = %s
                ORDER BY id
                LIMIT %s;
                """,
                (source_group, limit),
            )
        else:
            cur.execute(
                """
                SELECT id, chunk_text
                FROM chunks
                WHERE embedding IS NULL
                ORDER BY id
                LIMIT %s;
                """,
                (limit,),
            )

        return cur.fetchall()


def update_embeddings_bulk(
    conn,
    items: list[tuple[int, list[float]]],
) -> None:
    """
    Обновляем пачку embeddings.

    pgvector принимает строку вида '[0.1,0.2,...]'::vector.
    """

    if not items:
        return

    with conn.cursor() as cur:
        for chunk_id, embedding in items:
            embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"

            cur.execute(
                """
                UPDATE chunks
                SET embedding = %s::vector
                WHERE id = %s;
                """,
                (embedding_str, chunk_id),
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed chunks without embeddings using sentence-transformers"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"SentenceTransformer model name. Default: {DEFAULT_MODEL_NAME}",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Model encode batch size. Default: {DEFAULT_BATCH_SIZE}",
    )

    parser.add_argument(
        "--fetch-page-size",
        type=int,
        default=DEFAULT_FETCH_PAGE_SIZE,
        help=f"How many DB rows to fetch per outer loop. Default: {DEFAULT_FETCH_PAGE_SIZE}",
    )

    parser.add_argument(
        "--source-group",
        default=None,
        help="Optional filter by chunks.metadata->>'source_group', e.g. market_reports_cable",
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional maximum number of chunks to embed in this run.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    if args.fetch_page_size <= 0:
        raise ValueError("--fetch-page-size must be > 0")

    print("=" * 80)
    print("EMBED CHUNKS")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Fetch page size: {args.fetch_page_size}")
    print(f"Source group filter: {args.source_group}")
    print(f"Max chunks: {args.max_chunks}")
    print("=" * 80)

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    conn = get_conn()

    try:
        total_remaining = count_chunks_without_embeddings(
            conn=conn,
            source_group=args.source_group,
        )

        print(f"Chunks without embeddings at start: {total_remaining}")

        if total_remaining == 0:
            print("No chunks without embeddings found.")
            return

        total_done = 0

        while True:
            if args.max_chunks is not None:
                remaining_allowed = args.max_chunks - total_done
                if remaining_allowed <= 0:
                    print(f"Reached --max-chunks={args.max_chunks}")
                    break

                fetch_limit = min(args.fetch_page_size, remaining_allowed)
            else:
                fetch_limit = args.fetch_page_size

            rows = load_chunks_without_embeddings(
                conn=conn,
                limit=fetch_limit,
                source_group=args.source_group,
            )

            if not rows:
                break

            print(f"Fetched {len(rows)} chunks without embeddings...")

            for i in range(0, len(rows), args.batch_size):
                batch = rows[i:i + args.batch_size]
                chunk_ids = [row[0] for row in batch]
                texts = [row[1] for row in batch]

                embeddings = model.encode(
                    texts,
                    batch_size=args.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                update_items: list[tuple[int, list[float]]] = [
                    (chunk_id, embedding.tolist())
                    for chunk_id, embedding in zip(chunk_ids, embeddings)
                ]

                update_embeddings_bulk(conn, update_items)
                conn.commit()

                total_done += len(batch)
                print(f"Embedded total: {total_done}")

                if args.max_chunks is not None and total_done >= args.max_chunks:
                    print(f"Reached --max-chunks={args.max_chunks}")
                    break

        remaining_after = count_chunks_without_embeddings(
            conn=conn,
            source_group=args.source_group,
        )

        print("=" * 80)
        print("DONE")
        print("=" * 80)
        print(f"Total embedded this run: {total_done}")
        print(f"Chunks without embeddings remaining: {remaining_after}")
        print("=" * 80)

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()