from __future__ import annotations

from pathlib import Path

from .parsers.base import BaseParser


class ParserRegistry:
    """
    Реестр доступных парсеров.

    Нужен, чтобы pipeline не знал напрямую,
    какой класс использовать для PDF, DOCX, HTML и т.д.
    """

    def __init__(self) -> None:
        self._parsers: list[BaseParser] = []

    def register(self, parser: BaseParser) -> None:
        self._parsers.append(parser)

    def get_parser(self, path: Path) -> BaseParser:
        for parser in self._parsers:
            if parser.supports(path):
                return parser

        supported = sorted(
            {
                ext
                for parser in self._parsers
                for ext in parser.supported_extensions
            }
        )

        raise ValueError(
            f"No parser registered for file: {path}. "
            f"Supported extensions: {supported}"
        )

    def get_by_name(self, name: str) -> BaseParser:
        """
        Возвращает зарегистрированный парсер по его parser_name.

        Используется StrategyRouter / pipeline, когда стратегия
        однозначно указывает конкретный backend
        (например, "pymupdf_table_like" → парсер "pymupdf").
        """

        for parser in self._parsers:
            if parser.parser_name == name:
                return parser

        available = [p.parser_name for p in self._parsers]
        raise ValueError(
            f"No parser registered with name '{name}'. "
            f"Available: {available}"
        )

    def has(self, name: str) -> bool:
        return any(parser.parser_name == name for parser in self._parsers)

    def list_parsers(self) -> list[str]:
        return [
            f"{parser.parser_name} ({', '.join(sorted(parser.supported_extensions))})"
            for parser in self._parsers
        ]
