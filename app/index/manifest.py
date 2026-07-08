"""data/manifest.json: filename, sha256, pages, added_at.

Backs list/remove and ingest dedup, since a vector-store-only index has no
docstore to enumerate (see plan.md Review finding #3).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import config

_HASH_CHUNK_SIZE = 1 << 20  # 1 MiB, streamed so large PDFs aren't fully loaded to hash


@dataclass(frozen=True)
class ManifestEntry:
    source: str
    sha256: str
    pages: int
    added_at: str


def _path(manifest_path: Path | None) -> Path:
    return manifest_path or config.MANIFEST_PATH


def load_entries(manifest_path: Path | None = None) -> list[ManifestEntry]:
    path = _path(manifest_path)
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [ManifestEntry(**entry) for entry in raw.get("documents", [])]


def save_entries(entries: list[ManifestEntry], manifest_path: Path | None = None) -> None:
    """Write entries atomically: build in a temp file, then os.replace() it over
    the target so a crash/kill mid-write can't leave manifest.json truncated."""
    path = _path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"documents": [asdict(e) for e in entries]}, f, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def has_hash(sha256: str, manifest_path: Path | None = None) -> bool:
    return any(e.sha256 == sha256 for e in load_entries(manifest_path))


def find_by_source(entries: list[ManifestEntry], source: str) -> ManifestEntry | None:
    """Look up an entry by source filename in an already-loaded entry list."""
    return next((e for e in entries if e.source == source), None)


def add_entry(entry: ManifestEntry, manifest_path: Path | None = None) -> None:
    """Add entry, replacing any existing entry with the same source filename
    (so re-uploading a changed file under the same name updates, not duplicates)."""
    entries = [e for e in load_entries(manifest_path) if e.source != entry.source]
    entries.append(entry)
    save_entries(entries, manifest_path)


def remove_entry(source: str, manifest_path: Path | None = None) -> None:
    save_entries(
        [e for e in load_entries(manifest_path) if e.source != source], manifest_path
    )


def sha256_file(path: Path, chunk_size: int = _HASH_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
