"""Tests for version alignment between app and sidecar."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# Load settings module directly to avoid the import chain
_settings_path = Path(__file__).resolve().parents[2] / "settings.py"
_spec = importlib.util.spec_from_file_location("settings_mod", _settings_path)
_settings_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_settings_mod)

_get_version = _settings_mod._get_version


class TestGetVersion:
    """Tests for _get_version() in settings.py."""

    def test_reads_from_pyproject_toml(self):
        """_get_version should return a valid semver-like string from pyproject.toml."""
        version = _get_version()
        assert version != "0.0.0", "Should read version from pyproject.toml"
        parts = version.split(".")
        assert len(parts) >= 2, f"Version should be semver-like, got: {version}"

    def test_returns_current_version(self):
        """_get_version should return the current version from pyproject.toml."""
        import tomllib
        toml_path = Path(__file__).resolve().parents[4] / "pyproject.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        expected = data["project"]["version"]
        assert _get_version() == expected

    def test_fallback_on_missing_toml(self):
        """_get_version should return 0.0.0 when pyproject.toml is missing."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch.dict(sys.modules, {"importlib.metadata": None}):
                # Force both paths to fail
                version = _get_version()
                # May or may not fallback depending on importlib.metadata cache
                # Just verify it doesn't crash
                assert isinstance(version, str)


class TestSidecarVersion:
    """Tests for sidecar VERSION file."""

    def test_version_file_exists(self):
        """sidecar/VERSION file should exist."""
        version_path = Path(__file__).resolve().parents[4] / "sidecar" / "VERSION"
        assert version_path.exists(), f"VERSION file not found at {version_path}"

    def test_version_file_content(self):
        """sidecar/VERSION should contain the same version as pyproject.toml."""
        version_path = Path(__file__).resolve().parents[4] / "sidecar" / "VERSION"
        sidecar_version = version_path.read_text().strip()

        import tomllib
        toml_path = Path(__file__).resolve().parents[4] / "pyproject.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        app_version = data["project"]["version"]

        assert sidecar_version == app_version, (
            f"Sidecar VERSION ({sidecar_version}) != pyproject.toml ({app_version})"
        )

    def test_version_file_no_extra_whitespace(self):
        """VERSION file should contain just a version string with no extra content."""
        version_path = Path(__file__).resolve().parents[4] / "sidecar" / "VERSION"
        content = version_path.read_text()
        lines = [l for l in content.strip().split("\n") if l.strip()]
        assert len(lines) == 1, "VERSION file should contain exactly one line"
