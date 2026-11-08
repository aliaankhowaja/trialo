"""Configuration loading and small helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when a run configuration is malformed."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ConfigError(f"config must be a mapping: {config_path}")
    for section in ("model", "data", "lora", "train", "eval"):
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"config is missing mapping section {section!r}")
    return config


def config_hash(config: dict[str, Any]) -> str:
    """Return a stable SHA256 hash of the run configuration."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def get_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a mapping section, raising a clear error if absent."""
    value = config.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"config section {name!r} is not a mapping")
    return value
