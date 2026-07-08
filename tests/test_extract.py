"""Tests for app/ingest/extract.py (T1.2, T1.4-T1.7)."""
from pathlib import Path

import pytest
from llama_index.core import Document

from app.ingest.extract import ExtractionError, extract_documents

FIXTURES = Path(__file__).parent / "fixtures"


def test_born_digital_pdf_returns_direct_text_with_page_numbers():
    # Arrange
    path = FIXTURES / "born_digital.pdf"

    # Act
    docs = extract_documents(path)

    # Assert
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert "mitochondria" in docs[0].text.lower()
    assert docs[0].metadata["page"] == 1
    assert "photosynthesis" in docs[1].text.lower()
    assert docs[1].metadata["page"] == 2


def test_born_digital_pdf_sets_source_metadata():
    # Arrange
    path = FIXTURES / "born_digital.pdf"

    # Act
    docs = extract_documents(path)

    # Assert
    assert all(d.metadata["source"] == "born_digital.pdf" for d in docs)


def test_scanned_pdf_routes_to_ocr_and_retains_page_number():
    # Arrange
    path = FIXTURES / "scanned.pdf"

    # Act
    docs = extract_documents(path)

    # Assert
    assert len(docs) == 1
    assert "second law" in docs[0].text.lower()
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["source"] == "scanned.pdf"


def test_image_file_returns_single_document():
    # Arrange
    path = FIXTURES / "note.png"

    # Act
    docs = extract_documents(path)

    # Assert
    assert len(docs) == 1
    assert "boiling point of water" in docs[0].text.lower()
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["source"] == "note.png"


def test_corrupt_pdf_raises_extraction_error(tmp_path):
    # Arrange
    bad_pdf = tmp_path / "garbage.pdf"
    bad_pdf.write_bytes(b"not a real pdf")

    # Act / Assert
    with pytest.raises(ExtractionError):
        extract_documents(bad_pdf)


def test_empty_file_raises_extraction_error(tmp_path):
    # Arrange
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")

    # Act / Assert
    with pytest.raises(ExtractionError):
        extract_documents(empty_pdf)


def test_unsupported_file_type_raises_extraction_error(tmp_path):
    # Arrange
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("hello")

    # Act / Assert
    with pytest.raises(ExtractionError):
        extract_documents(txt_file)


def test_missing_file_raises_extraction_error(tmp_path):
    # Arrange
    missing = tmp_path / "does_not_exist.pdf"

    # Act / Assert
    with pytest.raises(ExtractionError):
        extract_documents(missing)
