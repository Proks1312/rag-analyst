from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> None:
    print("=" * 80)
    print("RUN")
    print("=" * 80)
    print(" ".join(command))

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(command)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run modular ingestion for all PDFs in a folder"
    )

    parser.add_argument(
        "--input-dir",
        default="data/raw",
        help="Folder with source PDF files",
    )

    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="File pattern, default: *.pdf",
    )

    parser.add_argument(
        "--project",
        default="RAG Analyst",
        help="Project metadata",
    )

    parser.add_argument(
        "--market",
        default=None,
        help="Market metadata",
    )

    parser.add_argument(
        "--product",
        default=None,
        help="Product metadata",
    )

    parser.add_argument(
        "--language",
        default="unknown",
        help="Language metadata: ru/en/zh/mixed/unknown",
    )

    parser.add_argument(
        "--skip-db-load",
        action="store_true",
        help="Only create processed files, do not load rag_chunks to Postgres",
    )

    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Do not run embed_chunks.py after DB load",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    input_dir = project_root / args.input_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    pdf_files = sorted(input_dir.glob(args.pattern))

    if not pdf_files:
        print(f"No files found: {input_dir / args.pattern}")
        return

    print("=" * 80)
    print("BATCH INGESTION")
    print("=" * 80)
    print(f"Input dir: {input_dir}")
    print(f"Files: {len(pdf_files)}")
    print(f"Project: {args.project}")
    print(f"Market: {args.market}")
    print(f"Product: {args.product}")
    print(f"Language: {args.language}")
    print("=" * 80)

    python_exe = sys.executable

    for pdf_path in pdf_files:
        print("\n" + "#" * 80)
        print(f"PROCESS FILE: {pdf_path.name}")
        print("#" * 80)

        # ВАЖНО:
        # Этот скрипт предполагает, что run_ingestion_test.py уже умеет принимать параметры.
        # Если он пока хардкодит test.pdf, на следующем шаге сделаем run_ingestion_file.py.
        command = [
            python_exe,
            "run_ingestion_file.py",
            "--pdf",
            str(pdf_path),
            "--project",
            args.project,
            "--language",
            args.language,
        ]

        if args.market:
            command.extend(["--market", args.market])

        if args.product:
            command.extend(["--product", args.product])

        run_command(command)

        rag_chunks_path = project_root / "data" / "processed" / "rag_chunks" / f"{pdf_path.stem}.rag_chunks.jsonl"

        if not args.skip_db_load:
            run_command(
                [
                    python_exe,
                    "load_rag_chunks_to_db.py",
                    "--rag-chunks",
                    str(rag_chunks_path),
                ]
            )

    if not args.skip_db_load and not args.skip_embeddings:
        run_command(
            [
                python_exe,
                "embed_chunks.py",
            ]
        )

    print("=" * 80)
    print("BATCH DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()