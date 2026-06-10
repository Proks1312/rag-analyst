from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz  # PyMuPDF


def safe_stem(name: str) -> str:
    """
    Делает безопасное ASCII-имя файла.
    Используется только если output_path явно не передан.
    """
    name = name.lower()
    name = name.replace("ё", "e")

    translit = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e",
        "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l",
        "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s",
        "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch",
        "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e",
        "ю": "yu", "я": "ya",
    }

    result = []
    for ch in name:
        result.append(translit.get(ch, ch))

    name = "".join(result)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")

    return name or "document"


def repair_pdf(
    input_path: Path,
    output_dir: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Repair / normalize PDF через PyMuPDF.

    Поддерживает два режима:

    1. Точный output:
       repair_pdf(input_path, output_path=Path(".../file.repaired.pdf"))

    2. Output folder:
       repair_pdf(input_path, output_dir=Path("data/raw_fixed"))
       Тогда имя будет создано через safe_stem().
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"PDF not found: {input_path}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if output_dir is None:
            output_dir = Path("data/raw_fixed")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{safe_stem(input_path.stem)}.pdf"

    print("=" * 80)
    print("REPAIR PDF")
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    orig = fitz.open(input_path)
    print(f"Pages (raw): {orig.page_count}")

    # Собираем чистый документ только из валидных, непустых страниц.
    # Это нужно для PDF, прошедших через iLovePDF / Контур.Фокус, где
    # /Pages может содержать «фантомные» страницы (например, /Count=114,
    # а реально читаются 20). Простой save с garbage=4 такие фантомы
    # сохранит, и Docling потом OOM-ится на пустых страницах.
    new_doc = fitz.open()
    kept = 0
    for i in range(orig.page_count):
        try:
            page = orig[i]
        except Exception:
            continue

        try:
            has_text = bool(page.get_text("text").strip())
        except Exception:
            has_text = False
        try:
            has_images = bool(page.get_images(full=True))
        except Exception:
            has_images = False
        try:
            has_drawings = bool(page.get_drawings())
        except Exception:
            has_drawings = False

        if not (has_text or has_images or has_drawings):
            continue

        try:
            new_doc.insert_pdf(orig, from_page=i, to_page=i)
            kept += 1
        except Exception:
            continue

    print(f"Pages kept: {kept} (dropped {orig.page_count - kept} blank/invalid)")

    if kept == 0:
        new_doc.close()
        orig.close()
        raise RuntimeError(
            "Не осталось валидных страниц после фильтрации — "
            "PDF полностью битый или пустой."
        )

    new_doc.save(
        output_path,
        garbage=4,
        deflate=True,
        clean=True,
    )

    new_doc.close()
    orig.close()

    print("Done.")

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Repair/normalize PDF via PyMuPDF")

    parser.add_argument(
        "--input",
        required=True,
        help="Input PDF path",
    )

    parser.add_argument(
        "--output-dir",
        default="data/raw_fixed",
        help="Output directory for repaired PDF. Used only if --output is not provided.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Exact output PDF path. If provided, --output-dir is ignored.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repair_pdf(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        output_path=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()