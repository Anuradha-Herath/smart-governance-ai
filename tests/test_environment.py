"""Smoke test to verify environment, directory structure, and dependencies."""

from __future__ import annotations

import sys
from pathlib import Path


def test_python_version() -> None:
    """Ensure Python version is 3.10 or newer."""
    assert sys.version_info >= (3, 10), f"Python version must be >= 3.10, got {sys.version}"


def test_directory_structure() -> None:
    """Ensure all required root directories exist."""
    project_root = Path(__file__).resolve().parent.parent
    expected_dirs = [
        "configs",
        "data",
        "src",
        "notebooks",
        "annotations",
        "models",
        "outputs",
        "tests",
        "app",
        "docs",
    ]
    for d in expected_dirs:
        dir_path = project_root / d
        assert dir_path.is_dir(), f"Missing required directory: {d}"


def test_dataset_register_presence() -> None:
    """Verify the Datasets folder and dataset_register.csv exist adjacent to the project root."""
    project_root = Path(__file__).resolve().parent.parent
    register_path = project_root.parent / "Datasets" / "dataset_register.csv"
    assert register_path.is_file(), f"Dataset register not found at {register_path}"
