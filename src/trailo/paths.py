"""Repository-relative paths used by the pipeline."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
DEFAULT_MANIFEST = SPLITS_DIR / "split_manifest.json"
LOGS_DIR = Path(__file__).resolve().parent / "logs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def repo_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository root unless absolute."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path
