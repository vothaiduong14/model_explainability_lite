"""I/O helpers for explainability runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import polars as pl
import yaml


def generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("explain_%Y%m%d_%H%M%S")


def ensure_run_dirs(output_root: Path, run_id: str) -> dict[str, Path]:
    base = output_root / run_id
    paths = {
        "base": base,
        "plots": base / "plots",
        "logs": base / "logs",
        "reports": base / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def write_validation_checks(path: Path, checks: Iterable[dict]) -> None:
    pl.DataFrame(list(checks)).write_csv(path)


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_dataset(path: Path) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pl.scan_csv(path)
    raise ValueError(f"Unsupported scoring dataset format: {path.suffix}")
