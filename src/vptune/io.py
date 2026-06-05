"""Schema-versioned JSON IO."""

import json
import os
import uuid
from pathlib import Path
from typing import Any

from vptune.identities import to_json_value
from vptune.schemas import validate_json_record


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write canonical JSON to a path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_json_path(path)

    try:
        temporary_path.write_text(_canonical_json_text(payload), encoding="utf-8")
        Path(temporary_path).replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Write canonical JSON only when the path does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_json_path(path)

    try:
        temporary_path.write_text(_canonical_json_text(payload), encoding="utf-8")
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _canonical_json_text(payload: dict[str, Any]) -> str:
    json_value = to_json_value(payload)
    text = json.dumps(json_value, sort_keys=True, indent=2)

    return f"{text}\n"


def _temporary_json_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from a path.

    Returns:
        Parsed JSON object.

    Raises:
        TypeError: If the JSON value is not an object.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        message = f"JSON file does not contain an object: {path}"
        raise TypeError(message)

    return payload


def write_record(path: Path, payload: dict[str, Any]) -> None:
    """Validate and write a saved record."""
    validate_json_record(payload)
    write_json(path, payload)


def write_record_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Validate and write a saved record only when the path does not exist."""
    validate_json_record(payload)
    write_json_exclusive(path, payload)


def read_record(path: Path) -> dict[str, Any]:
    """Read and validate a saved record.

    Returns:
        Parsed saved record.
    """
    payload = read_json(path)
    validate_json_record(payload)

    return payload
