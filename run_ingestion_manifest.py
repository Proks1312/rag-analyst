from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any


# ============================================================
# COMMAND RUNNER
# ============================================================

def run_command(command: list[str], allow_fail: bool = False) -> bool:
    print("=" * 80)
    print("RUN")
    print("=" * 80)
    print(" ".join(f'"{x}"' if " " in x else x for x in command))

    result = subprocess.run(command)

    if result.returncode != 0:
        message = f"Command failed with code {result.returncode}: {' '.join(command)}"

        if allow_fail:
            print("=" * 80)
            print("COMMAND FAILED, SKIPPING")
            print("=" * 80)
            print(message)
            return False

        raise RuntimeError(message)

    return True


# ============================================================
# MANIFEST READER
# ============================================================

REQUIRED_COLUMNS = {
    "file_path",
    "project",
    "market",
    "product",
    "language",
}

OPTIONAL_COLUMNS = {
    "source_group",
    "source_type",
    "topic",
}


def detect_csv_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:4096]

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        return dialect.delimiter
    except Exception:
        comma_count = sample.count(",")
        semicolon_count = sample.count(";")
        return ";" if semicolon_count > comma_count else ","


def normalize_header(value: str) -> str:
    return value.strip().replace("\ufeff", "")


def normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def read_manifest(path: Path) -> list[dict[str, str | None]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    delimiter = detect_csv_delimiter(path)

    print("=" * 80)
    print("READ MANIFEST")
    print("=" * 80)
    print(f"Manifest: {path}")
    print(f"Detected delimiter: {repr(delimiter)}")

    rows: list[dict[str, str | None]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {path}")

        fieldnames = [normalize_header(x) for x in reader.fieldnames]
        reader.fieldnames = fieldnames

        print(f"Columns: {fieldnames}")

        missing = REQUIRED_COLUMNS - set(fieldnames)

        if missing:
            raise ValueError(
                f"Manifest missing columns: {sorted(missing)}. "
                f"Required columns: {sorted(REQUIRED_COLUMNS)}"
            )

        present_optional = OPTIONAL_COLUMNS & set(fieldnames)
        if present_optional:
            print(f"Optional metadata columns found: {sorted(present_optional)}")

        for row_number, row in enumerate(reader, start=2):
            file_path = normalize_cell(row.get("file_path"))

            if not file_path:
                print(f"[WARN] Empty file_path at row {row_number}, skipping")
                continue

            project = normalize_cell(row.get("project")) or "RAG Analyst"
            market = normalize_cell(row.get("market")) or None
            product = normalize_cell(row.get("product")) or None
            language = normalize_cell(row.get("language")) or "unknown"

            source_group = normalize_cell(row.get("source_group")) or "general"
            source_type = normalize_cell(row.get("source_type")) or "pdf"
            topic = normalize_cell(row.get("topic")) or None

            rows.append(
                {
                    "file_path": file_path,
                    "project": project,
                    "market": market,
                    "product": product,
                    "language": language,
                    "source_group": source_group,
                    "source_type": source_type,
                    "topic": topic,
                    "_row_number": str(row_number),
                }
            )

    print(f"Rows loaded: {len(rows)}")
    print("=" * 80)

    return rows


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_file_path(project_root: Path, file_path_from_manifest: str) -> Path:
    raw = file_path_from_manifest.strip().strip('"').strip("'")
    path = Path(raw)

    if path.is_absolute():
        return path

    return project_root / path


def print_manifest_row(row: dict[str, str | None], file_path: Path) -> None:
    print("Manifest row:")
    print(f"  row_number:   {row.get('_row_number')}")
    print(f"  file_path:    {row.get('file_path')}")
    print(f"  resolved:     {file_path}")
    print(f"  project:      {row.get('project')}")
    print(f"  market:       {row.get('market')}")
    print(f"  product:      {row.get('product')}")
    print(f"  language:     {row.get('language')}")
    print(f"  source_group: {row.get('source_group')}")
    print(f"  source_type:  {row.get('source_type')}")
    print(f"  topic:        {row.get('topic')}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run modular ingestion from CSV manifest"
    )

    parser.add_argument(
        "--manifest",
        default="data/manifests/all_pdf_context_manifest.csv",
        help="CSV manifest path",
    )

    parser.add_argument(
        "--skip-db-load",
        action="store_true",
        help="Deprecated here. DB load is handled by run_full_pipeline step 3.",
    )

    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Deprecated here. Embeddings are handled by run_full_pipeline step 4.",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop whole batch if one file fails",
    )

    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Only profile all documents from manifest, do not parse them",
    )

    parser.add_argument(
        "--parser",
        default="auto",
        choices=[
            "auto",
            "docling",
            "pymupdf",
            "hybrid",
            "vlm",
            "profile",
            "docling_full",
            "pymupdf_text",
            "pymupdf_table_like",
            "hybrid_docling_then_pymupdf",
            "ocr_required",
        ],
        help="Parser/strategy mode for all documents. Default: auto (профилировщик выбирает сам)",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    manifest_path = Path(args.manifest)

    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path

    rows = read_manifest(manifest_path)

    if not rows:
        print(f"No rows in manifest: {manifest_path}")
        return

    print("=" * 80)
    print("MANIFEST INGESTION")
    print("=" * 80)
    print(f"Manifest: {manifest_path}")
    print(f"Files: {len(rows)}")
    print(f"Stop on error: {args.stop_on_error}")
    print(f"Profile only: {args.profile_only}")
    print(f"Parser/strategy mode: {args.parser}")
    print("=" * 80)

    python_exe = sys.executable

    processed_files: list[str] = []
    skipped_files: list[str] = []
    profiled_files: list[str] = []

    for row in rows:
        file_path = resolve_file_path(project_root, str(row["file_path"]))

        print("\n" + "#" * 80)
        print(f"PROCESS FILE: {file_path.name}")
        print("#" * 80)
        print_manifest_row(row, file_path)

        if not file_path.exists():
            message = f"PDF not found: {file_path}"

            if args.stop_on_error:
                raise FileNotFoundError(message)

            print("=" * 80)
            print("FILE NOT FOUND, SKIPPING")
            print("=" * 80)
            print(message)

            skipped_files.append(str(file_path))
            continue

        command = [
            python_exe,
            "run_ingestion_file.py",
            "--pdf",
            str(file_path),
            "--project",
            str(row["project"] or "RAG Analyst"),
            "--language",
            str(row["language"] or "unknown"),
            "--parser",
            args.parser,
        ]

        if row.get("market"):
            command.extend(["--market", str(row["market"])])

        if row.get("product"):
            command.extend(["--product", str(row["product"])])

        if row.get("source_group"):
            command.extend(["--source-group", str(row["source_group"])])

        if row.get("source_type"):
            command.extend(["--source-type", str(row["source_type"])])

        if row.get("topic"):
            command.extend(["--topic", str(row["topic"])])

        if args.profile_only:
            command.append("--profile-only")

        ok = run_command(
            command,
            allow_fail=not args.stop_on_error,
        )

        if not ok:
            print(f"SKIPPED FILE DUE TO INGESTION ERROR: {file_path.name}")
            skipped_files.append(str(file_path))
            continue

        if args.profile_only:
            profiled_files.append(str(file_path))
            continue

        # ВАЖНО:
        # Здесь НЕ проверяем rag_chunks.
        # STEP 1 должен создать embedding_records.
        # rag_chunks создаются отдельным STEP 2 через build_rag_chunks_from_records.py.
        embedding_records_path = (
            project_root
            / "data"
            / "processed"
            / "embedding_input"
            / f"{file_path.stem}.embedding_records.jsonl"
        )

        if not embedding_records_path.exists():
            message = f"Embedding records file not found after ingestion: {embedding_records_path}"

            if args.stop_on_error:
                raise FileNotFoundError(message)

            print("=" * 80)
            print("EMBEDDING RECORDS NOT FOUND, SKIPPING")
            print("=" * 80)
            print(message)

            skipped_files.append(str(file_path))
            continue

        processed_files.append(str(file_path))

    print("=" * 80)
    print("MANIFEST INGESTION DONE")
    print("=" * 80)

    if args.profile_only:
        print(f"Profiled files: {len(profiled_files)}")
        for file in profiled_files:
            print(f"  PROFILED: {file}")

    print(f"Processed files: {len(processed_files)}")
    for file in processed_files:
        print(f"  OK: {file}")

    print(f"Skipped files: {len(skipped_files)}")
    for file in skipped_files:
        print(f"  SKIPPED: {file}")

    print("=" * 80)


if __name__ == "__main__":
    main()