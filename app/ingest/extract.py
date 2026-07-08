"""Turn an uploaded PDF or image into Documents with {source, page} metadata.

Born-digital PDF pages use PyMuPDF's direct text extraction; pages with no
(or very little) extractable text are treated as scanned and routed through
OCR. Standalone images are OCR'd directly as a single page.
"""
import io
from pathlib import Path
from typing import Union

import fitz
import numpy as np
from llama_index.core import Document
from PIL import Image

from app.ingest.ocr import extract_text_from_image

# Below this many characters, a PDF page is treated as scanned (image-only)
# and rendered + OCR'd instead of trusting the (near-empty) text layer.
SCANNED_PAGE_TEXT_THRESHOLD = 10
OCR_RENDER_DPI = 200

PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


class ExtractionError(Exception):
    """Raised when a file can't be turned into Documents (corrupt, empty, unsupported)."""


def extract_documents(path: Union[str, Path]) -> list[Document]:
    """Dispatch a file to PDF or image extraction based on its extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    if not path.exists():
        raise ExtractionError(f"File not found: {path}")

    if suffix in PDF_SUFFIXES:
        return _extract_pdf(path)
    if suffix in IMAGE_SUFFIXES:
        return _extract_image(path)
    raise ExtractionError(f"Unsupported file type: {suffix or '(none)'}")


def _extract_pdf(path: Path) -> list[Document]:
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise ExtractionError(f"Could not open PDF {path.name}: {exc}") from exc

    try:
        if doc.page_count == 0:
            raise ExtractionError(f"PDF has no pages: {path.name}")

        documents = []
        for page_index, page in enumerate(doc):
            text = page.get_text("text").strip()
            if len(text) < SCANNED_PAGE_TEXT_THRESHOLD:
                pixmap = page.get_pixmap(dpi=OCR_RENDER_DPI)
                image = pixmap.tobytes("png")
                try:
                    text = extract_text_from_image(_png_bytes_to_array(image))
                except Exception as exc:
                    raise ExtractionError(
                        f"Could not OCR page {page_index + 1} of {path.name}: {exc}"
                    ) from exc
            documents.append(
                Document(text=text, metadata={"source": path.name, "page": page_index + 1})
            )
        return documents
    finally:
        doc.close()


def _extract_image(path: Path) -> list[Document]:
    try:
        text = extract_text_from_image(path)
    except Exception as exc:
        raise ExtractionError(f"Could not OCR image {path.name}: {exc}") from exc
    return [Document(text=text, metadata={"source": path.name, "page": 1})]


def _png_bytes_to_array(png_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return np.array(image)
