"""RapidOCR wrapper: image (or rendered PDF page) -> text."""
from pathlib import Path
from typing import Union

import numpy as np
from rapidocr import RapidOCR

ImageInput = Union[str, Path, np.ndarray]

_engine: RapidOCR | None = None


def _get_engine() -> RapidOCR:
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


def extract_text_from_image(image: ImageInput) -> str:
    """Run OCR on an image path or array, returning concatenated line text.

    Returns an empty string when no text is detected (blank/non-text image),
    rather than raising.
    """
    result = _get_engine()(image)
    if result is None or not result.txts:
        return ""
    return "\n".join(result.txts)
