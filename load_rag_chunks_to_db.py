from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json


load_dotenv()


PIPELINE_NAME = "modular_ingestion_v1"


def get_conn():
    return psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
    )


# ============================================================
# READ / HASH
# ============================================================

def read_rag_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"RAG chunks file not found: {path}")

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}, line {line_number}: {exc}"
                ) from exc

    return rows


def source_meta(chunk: dict[str, Any]) -> dict[str, Any]:
    meta = chunk.get("metadata") or {}
    if not isinstance(meta, dict):
        return {}
    return meta


def get_source_group(chunk: dict[str, Any]) -> str | None:
    meta = source_meta(chunk)
    return meta.get("source_group")


def get_topic(chunk: dict[str, Any]) -> str | None:
    meta = source_meta(chunk)
    return meta.get("topic")


def get_manifest_source_type(chunk: dict[str, Any]) -> str | None:
    meta = source_meta(chunk)
    return meta.get("manifest_source_type")


def build_chunk_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    Metadata, которая попадет в chunks.metadata JSONB.

    Важное:
    - source_metadata содержит полную metadata из rag_chunks.jsonl;
    - source_group/topic/manifest_source_type вынесены наверх для удобной фильтрации.
    """

    meta = source_meta(chunk)

    return {
        "pipeline": PIPELINE_NAME,

        "chunk_id": chunk.get("chunk_id"),
        "chunk_type": chunk.get("chunk_type"),

        "section_title": chunk.get("section_title"),
        "page_numbers": chunk.get("page_numbers", []),
        "source_record_ids": chunk.get("source_record_ids", []),

        "source_file": chunk.get("source_file"),
        "source_type": chunk.get("source_type"),
        "manifest_source_type": meta.get("manifest_source_type"),

        "source_group": meta.get("source_group"),
        "topic": meta.get("topic"),

        "project": chunk.get("project"),
        "market": chunk.get("market"),
        "product": chunk.get("product"),
        "language": chunk.get("language"),

        "selected_strategy": meta.get("selected_strategy"),
        "parser_backend": meta.get("parser_backend"),
        "fallback_parser": meta.get("fallback_parser"),
        "pdf_repaired": meta.get("pdf_repaired"),

        "original_source_path": meta.get("original_source_path"),
        "repaired_source_path": meta.get("repaired_source_path"),
        "source_path": meta.get("source_path"),

        "source_metadata": meta,
    }


def normalize_for_hash(value: Any) -> Any:
    """
    Приводим значение к стабильному виду для hash.

    Важно:
    - embedding в hash не включаем;
    - порядок ключей стабилизируется через json.dumps(sort_keys=True);
    - None оставляем как None.
    """
    if isinstance(value, dict):
        return {str(k): normalize_for_hash(v) for k, v in sorted(value.items())}

    if isinstance(value, list):
        return [normalize_for_hash(x) for x in value]

    return value


def chunk_signature_from_jsonl(chunk: dict[str, Any], chunk_index: int) -> dict[str, Any]:
    return {
        "chunk_index": chunk_index,
        "chunk_text": chunk.get("text"),
        "page": chunk.get("page_number"),
        "metadata": build_chunk_metadata(chunk),
    }


def chunk_signature_from_db(row: tuple[Any, ...]) -> dict[str, Any]:
    chunk_index, chunk_text, page, metadata = row

    if metadata is None:
        metadata = {}

    return {
        "chunk_index": chunk_index,
        "chunk_text": chunk_text,
        "page": page,
        "metadata": metadata,
    }


def compute_hash_from_signatures(signatures: list[dict[str, Any]]) -> str:
    normalized = normalize_for_hash(signatures)

    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_jsonl_chunks_hash(chunks: list[dict[str, Any]]) -> str:
    signatures = [
        chunk_signature_from_jsonl(chunk, chunk_index=idx)
        for idx, chunk in enumerate(chunks)
    ]

    return compute_hash_from_signatures(signatures)


def compute_db_document_chunks_hash(conn, document_id: int) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                chunk_index,
                chunk_text,
                page,
                metadata
            FROM chunks
            WHERE document_id = %s
            ORDER BY chunk_index;
            """,
            (document_id,),
        )

        rows = cur.fetchall()

    signatures = [chunk_signature_from_db(row) for row in rows]

    return compute_hash_from_signatures(signatures)


# ============================================================
# DB HELPERS
# ============================================================

def find_existing_documents(conn, file_name: str) -> list[int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id
            FROM documents
            WHERE file_name = %s
            ORDER BY id;
            """,
            (file_name,),
        )

        return [row[0] for row in cur.fetchall()]


def delete_existing_document_data(conn, file_name: str) -> list[int]:
    document_ids = find_existing_documents(conn, file_name)

    if not document_ids:
        print(f"No old documents found for file_name={file_name}")
        return []

    with conn.cursor() as cur:
        for document_id in document_ids:
            cur.execute(
                """
                DELETE FROM extracted_tables
                WHERE document_id = %s;
                """,
                (document_id,),
            )

            cur.execute(
                """
                DELETE FROM chunks
                WHERE document_id = %s;
                """,
                (document_id,),
            )

            cur.execute(
                """
                DELETE FROM documents
                WHERE id = %s;
                """,
                (document_id,),
            )

    print(f"Deleted old document data: {document_ids}")
    return document_ids


def insert_document(
    conn,
    first_chunk: dict[str, Any],
    rag_chunks_path: Path,
    rag_chunks_hash: str,
) -> int:
    source_file = first_chunk["source_file"]
    source_type = first_chunk.get("source_type", "unknown")
    project = first_chunk.get("project")
    market = first_chunk.get("market")
    product = first_chunk.get("product")
    language = first_chunk.get("language", "unknown")

    source_group = get_source_group(first_chunk)
    topic = get_topic(first_chunk)
    manifest_source_type = get_manifest_source_type(first_chunk)

    title_parts = [
        source_file,
        f"group={source_group}" if source_group else None,
        f"topic={topic}" if topic else None,
        f"project={project}" if project else None,
        f"market={market}" if market else None,
        f"product={product}" if product else None,
        f"manifest_type={manifest_source_type}" if manifest_source_type else None,
        f"pipeline={PIPELINE_NAME}",
        f"chunks_hash={rag_chunks_hash[:12]}",
    ]

    title = " | ".join(part for part in title_parts if part)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (
                file_name,
                file_path,
                source_type,
                title,
                language
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (
                source_file,
                str(rag_chunks_path),
                source_type,
                title,
                language,
            ),
        )

        return cur.fetchone()[0]


def insert_chunk(
    conn,
    document_id: int,
    chunk: dict[str, Any],
    chunk_index: int,
) -> None:
    metadata = build_chunk_metadata(chunk)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks (
                document_id,
                chunk_text,
                page,
                chunk_index,
                metadata,
                embedding
            )
            VALUES (%s, %s, %s, %s, %s, NULL);
            """,
            (
                document_id,
                chunk["text"],
                chunk.get("page_number"),
                chunk_index,
                Json(metadata),
            ),
        )


def existing_document_with_same_hash(
    conn,
    file_name: str,
    new_hash: str,
) -> int | None:
    document_ids = find_existing_documents(conn, file_name)

    if not document_ids:
        return None

    for document_id in document_ids:
        db_hash = compute_db_document_chunks_hash(conn, document_id)

        if db_hash == new_hash:
            return document_id

    return None


# ============================================================
# DEBUG / SUMMARY
# ============================================================

def print_chunk_summary(chunks: list[dict[str, Any]]) -> None:
    source_groups: dict[str, int] = {}
    topics: dict[str, int] = {}
    chunk_types: dict[str, int] = {}

    for chunk in chunks:
        sg = get_source_group(chunk) or "unknown"
        topic = get_topic(chunk) or "unknown"
        chunk_type = chunk.get("chunk_type") or "unknown"

        source_groups[sg] = source_groups.get(sg, 0) + 1
        topics[topic] = topics.get(topic, 0) + 1
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

    print("Chunk summary:")
    print(f"  source_groups: {source_groups}")
    print(f"  topics:        {topics}")
    print(f"  chunk_types:   {chunk_types}")


# ============================================================
# MAIN LOADER
# ============================================================

def load_rag_chunks_to_db(
    rag_chunks_path: Path,
    mode: str = "replace-changed",
    force: bool = False,
) -> None:
    chunks = read_rag_chunks(rag_chunks_path)

    if not chunks:
        print(f"No rag chunks found: {rag_chunks_path}")
        return

    source_file = chunks[0]["source_file"]
    rag_chunks_hash = compute_jsonl_chunks_hash(chunks)

    print("=" * 80)
    print("LOAD RAG CHUNKS TO DB")
    print("=" * 80)
    print(f"Input: {rag_chunks_path}")
    print(f"Source file: {source_file}")
    print(f"Chunks: {len(chunks)}")
    print(f"Mode: {mode}")
    print(f"Force: {force}")
    print(f"Chunks hash: {rag_chunks_hash}")
    print_chunk_summary(chunks)
    print("=" * 80)

    conn = get_conn()

    try:
        existing_ids = find_existing_documents(conn, source_file)

        print(f"Existing document ids: {existing_ids if existing_ids else 'none'}")

        if not force and mode != "append":
            same_document_id = existing_document_with_same_hash(
                conn=conn,
                file_name=source_file,
                new_hash=rag_chunks_hash,
            )

            if same_document_id is not None:
                print("=" * 80)
                print("SKIP")
                print("=" * 80)
                print(
                    "Document already loaded with the same chunks. "
                    "DB rows and embeddings were not touched."
                )
                print(f"Existing document_id: {same_document_id}")
                print("=" * 80)
                conn.rollback()
                return

        if mode == "skip-existing" and existing_ids and not force:
            print("=" * 80)
            print("SKIP")
            print("=" * 80)
            print(
                "Document already exists. "
                "Mode is skip-existing, so DB rows and embeddings were not touched."
            )
            print("=" * 80)
            conn.rollback()
            return

        if mode == "replace-changed" or force:
            if existing_ids:
                delete_existing_document_data(conn, source_file)

        elif mode == "append":
            print("Append mode: old documents will be kept.")

        else:
            raise ValueError(
                f"Unknown mode: {mode}. "
                "Expected: replace-changed, skip-existing, append"
            )

        document_id = insert_document(
            conn=conn,
            first_chunk=chunks[0],
            rag_chunks_path=rag_chunks_path,
            rag_chunks_hash=rag_chunks_hash,
        )

        for idx, chunk in enumerate(chunks):
            insert_chunk(
                conn=conn,
                document_id=document_id,
                chunk=chunk,
                chunk_index=idx,
            )

        conn.commit()

        print("=" * 80)
        print("DONE")
        print("=" * 80)
        print(f"Inserted document_id: {document_id}")
        print(f"Inserted chunks: {len(chunks)}")
        print("Embeddings are NULL only for inserted chunks. Run embed_chunks.py.")
        print("=" * 80)

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load modular RAG chunks JSONL to Postgres"
    )

    parser.add_argument(
        "--rag-chunks",
        required=True,
        help="Path to *.rag_chunks.jsonl",
    )

    parser.add_argument(
        "--mode",
        choices=["replace-changed", "skip-existing", "append"],
        default="replace-changed",
        help=(
            "replace-changed: skip if same chunks, replace if changed; "
            "skip-existing: skip if any document with same file_name exists; "
            "append: always add another document version"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force replace even if chunks hash is the same.",
    )

    # Совместимость со старым параметром.
    parser.add_argument(
        "--keep-old",
        action="store_true",
        help="Deprecated alias for --mode append.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mode = "append" if args.keep_old else args.mode

    load_rag_chunks_to_db(
        rag_chunks_path=Path(args.rag_chunks),
        mode=mode,
        force=args.force,
    )


if __name__ == "__main__":
    main()