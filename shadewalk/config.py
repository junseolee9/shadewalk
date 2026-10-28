"""Load config.yaml and resolve paths against the project root."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def contains(self, lon, lat):
        return (lon >= self.min_lon) & (lon <= self.max_lon) & (lat >= self.min_lat) & (lat <= self.max_lat)


class Config(dict):
    """Plain dict with a few helpers."""

    @property
    def bbox(self) -> BBox:
        return BBox(**self["bbox"])

    def path(self, *keys: str) -> Path:
        node: Any = self
        for k in keys:
            node = node[k]
        p = Path(node)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Read KEY=value lines from .env into the environment (existing variables win)."""
    import os

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_config(path: str | Path | None = None) -> Config:
    load_dotenv()
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
