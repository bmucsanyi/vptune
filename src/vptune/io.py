"""Schema-versioned JSON IO."""

import json
from pathlib import Path
from typing import Any

from vptune.identities import to_json_value
from vptune.schemas import validate_json_record


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write canonical JSON to a path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    json_value = to_json_value(payload)
    text = json.dumps(json_value, sort_keys=True, indent=2)
    path.write_text(f"{text}\n", encoding="utf-8")


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


def read_record(path: Path) -> dict[str, Any]:
    """Read and validate a saved record.

    Returns:
        Parsed saved record.
    """
    payload = read_json(path)
    validate_json_record(payload)

    return payload
