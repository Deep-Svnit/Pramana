from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz
from pdf2image import convert_from_path


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


def document_id_for(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", path.stem).strip("_")


def render_pdf_pages(pdf_path: Path, output_dir: Path, dpi: int = 200) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = convert_from_path(str(pdf_path), dpi=dpi)
    image_paths: list[Path] = []
    for index, image in enumerate(images, start=1):
        out = output_dir / f"page_{index:04d}.png"
        image.save(out)
        image_paths.append(out)
    return image_paths


def extract_pdf_text(pdf_path: Path) -> list[PageText]:
    doc = fitz.open(pdf_path)
    pages: list[PageText] = []
    for index, page in enumerate(doc, start=1):
        pages.append(PageText(page_number=index, text=page.get_text("text").strip()))
    return pages
