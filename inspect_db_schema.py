from __future__ import annotations

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_conn():
    return psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
    )


def print_columns(conn, table_name: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position;
            """,
            (table_name,),
        )

        rows = cur.fetchall()

    print("=" * 80)
    print(table_name)
    print("=" * 80)

    for column_name, data_type, is_nullable in rows:
        print(f"{column_name:25} {data_type:25} nullable={is_nullable}")


def main() -> None:
    conn = get_conn()

    try:
        print_columns(conn, "documents")
        print_columns(conn, "chunks")
        print_columns(conn, "extracted_tables")
    finally:
        conn.close()


if __name__ == "__main__":
    main()