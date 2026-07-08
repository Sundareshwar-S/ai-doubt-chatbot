"""Tests for app/ingest/ocr.py (T1.3)."""
from pathlib import Path

import pytest

from app.ingest.ocr import extract_text_from_image

FIXTURES = Path(__file__).parent / "fixtures"


def test_extract_text_from_image_returns_known_text():
    # Arrange
    image_path = FIXTURES / "note.png"

    # Act
    text = extract_text_from_image(image_path)

    # Assert
    assert "boiling point of water" in text.lower()


def test_extract_text_from_image_handles_blank_image(tmp_path):
    # Arrange
    from PIL import Image

    blank_path = tmp_path / "blank.png"
    Image.new("RGB", (100, 100), color="white").save(blank_path)

    # Act
    text = extract_text_from_image(blank_path)

    # Assert
    assert text == ""
