"""Domain exceptions -> structured `{error: {code, message}}` JSON (Phase 5).

Domain errors are plain exceptions (not Pydantic); the response body they
render to is Pydantic (`app/api/schemas/errors.py`) -- kept in separate files
per this project's FastAPI rule ("keep request/update/response schemas
separate"). `app.ingest.extract.ExtractionError` is a pre-existing Phase 1
exception, not a new domain error, so it gets its own handler rather than
being wrapped in `ApiError`.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.schemas.errors import ErrorDetail, ErrorResponse
from app.ingest.extract import ExtractionError


class ApiError(Exception):
    """Base class for domain errors that render as the structured error envelope."""

    status_code: int = 400
    code: str = "bad_request"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class UploadTooLargeError(ApiError):
    status_code = 413
    code = "upload_too_large"


class UnsupportedFileTypeError(ApiError):
    status_code = 400
    code = "unsupported_file_type"


class DocumentNotFoundError(ApiError):
    status_code = 404
    code = "document_not_found"


class OllamaUnavailableError(ApiError):
    status_code = 503
    code = "ollama_unavailable"


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(ExtractionError)
    async def handle_extraction_error(request: Request, exc: ExtractionError) -> JSONResponse:
        return _error_response(422, "extraction_failed", str(exc))
