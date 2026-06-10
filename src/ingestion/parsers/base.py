from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import DocumentMetadata, ParsedDocument


class BaseParser(ABC):
    """
    Базовый интерфейс для всех parser backend-ов.

    Любой конкретный парсер должен:
    1. Понимать, поддерживает ли он конкретный файл.
    2. Превращать файл в ParsedDocument.
    """

    parser_name: str = "base"
    parser_version: str = "0.1.0"
    supported_extensions: set[str] = set()

    def supports(self, path: Path) -> bool:
        """
        Проверяет, может ли парсер обработать файл по расширению.
        """
        return path.suffix.lower() in self.supported_extensions

    @abstractmethod
    def parse(
        self,
        path: Path,
        metadata: DocumentMetadata,
    ) -> ParsedDocument:
        """
        Основной метод парсера.

        На вход:
        - путь к файлу;
        - метаданные документа.

        На выход:
        - ParsedDocument с blocks и tables.
        """
        raise NotImplementedError