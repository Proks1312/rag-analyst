from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ingestion.models import ParsedDocument


class LocalJsonlStorage:
    """
    Локальное сохранение результата prechunking в JSONL.

    Для MVP сохраняем сущности раздельно:
    - blocks
    - tables
    - visuals
    """

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, document: ParsedDocument) -> Path:
        source_stem = Path(document.metadata.source_file).stem

        blocks_path = self.output_dir / f"{source_stem}.blocks.jsonl"
        tables_path = self.output_dir / f"{source_stem}.tables.jsonl"
        visuals_path = self.output_dir / f"{source_stem}.visuals.jsonl"

        self._save_blocks(document, blocks_path)
        self._save_tables(document, tables_path)
        self._save_visuals(document, visuals_path)

        return blocks_path

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _save_blocks(self, document: ParsedDocument, path: Path) -> None:
        rows = []
        for block in document.non_empty_blocks():
            rows.append(
                {
                    "entity_type": "block",
                    "document": asdict(document.metadata),
                    "entity": asdict(block),
                }
            )
        self._write_jsonl(path, rows)

    def _save_tables(self, document: ParsedDocument, path: Path) -> None:
        rows = []
        for table in document.non_empty_tables():
            rows.append(
                {
                    "entity_type": "table",
                    "document": asdict(document.metadata),
                    "entity": asdict(table),
                }
            )
        self._write_jsonl(path, rows)

    def _save_visuals(self, document: ParsedDocument, path: Path) -> None:
        rows = []
        for visual in document.non_empty_visuals():
            rows.append(
                {
                    "entity_type": "visual",
                    "document": asdict(document.metadata),
                    "entity": asdict(visual),
                }
            )
        self._write_jsonl(path, rows)