from __future__ import annotations

from ingestion.models import ParsedBlock, ParsedDocument, ParsedTable


class TableExtractor:
    """
    Переносит markdown-таблицы из ParsedBlock в ParsedTable.

    После этого:
    - обычные текстовые блоки остаются в document.blocks;
    - таблицы сохраняются отдельно в document.tables;
    - связь с документом, страницей, языком и metadata сохраняется.
    """

    def extract_tables(self, document: ParsedDocument) -> ParsedDocument:
        new_blocks: list[ParsedBlock] = []
        new_tables: list[ParsedTable] = list(document.tables)

        table_order = len(new_tables)

        for block in document.blocks:
            if block.block_type != "table":
                new_blocks.append(block)
                continue

            table = ParsedTable(
                markdown_table=block.block_text,
                raw_data=[],
                table_order=table_order,
                source_file=block.source_file,
                source_type=block.source_type,
                page_number=block.page_number,
                table_title=block.section_title,
                language=block.language,
                parser_name=block.parser_name,
                parser_version=block.parser_version,
                metadata={
                    **block.metadata,
                    "extracted_from_block_order": block.block_order,
                    "source_block_type": block.block_type,
                    "table_format": "markdown",
                },
            )

            new_tables.append(table)
            table_order += 1

        document.blocks = new_blocks
        document.tables = new_tables

        return document