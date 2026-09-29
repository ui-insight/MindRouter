"""The dependency lock stays in force and stays coherent with pyproject.toml.

Background: pyproject pins are floors, so a rebuild resolves the newest
release of everything. On 2026-09-29 that pulled in SQLAlchemy 2.1.1, which
dropped greenlet, and the async engine could not import on the new image.
constraints.txt turns the floors into exact versions and the Dockerfile
installs against it. These tests fail if either half of that quietly stops
being true.
"""

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[4]
CONSTRAINTS = ROOT / "constraints.txt"
PYPROJECT = ROOT / "pyproject.toml"
DOCKERFILE = ROOT / "Dockerfile"


def _pins() -> dict[str, Version]:
    pins: dict[str, Version] = {}
    for raw in CONSTRAINTS.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" in line and not line.startswith("-"), f"not an exact pin: {line!r}"
        name, version = line.split("==", 1)
        assert "@" not in version and "+" not in version, f"unpublishable pin: {line!r}"
        pins[canonicalize_name(name)] = Version(version)
    return pins


def _declared() -> list[Requirement]:
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    reqs = [Requirement(r) for r in project["dependencies"]]
    # The image installs the saml extra (Dockerfile: pip install -e .[saml]).
    reqs += [Requirement(r) for r in project.get("optional-dependencies", {}).get("saml", [])]
    return reqs


class TestLockFile:
    def test_every_declared_dependency_is_pinned(self):
        pins = _pins()
        missing = [r.name for r in _declared() if canonicalize_name(r.name) not in pins]
        assert not missing, f"declared in pyproject but absent from constraints.txt: {missing}"

    def test_every_pin_satisfies_its_pyproject_specifier(self):
        pins = _pins()
        bad = []
        for r in _declared():
            pinned = pins[canonicalize_name(r.name)]
            if r.specifier and not r.specifier.contains(pinned, prereleases=True):
                bad.append(f"{r.name}: pinned {pinned}, pyproject wants {r.specifier}")
        assert not bad, "\n".join(bad)

    def test_sqlalchemy_stays_on_a_greenlet_bearing_line(self):
        # Until the project depends on sqlalchemy[asyncio], 2.1+ silently
        # loses greenlet and the async engine cannot import.
        pins = _pins()
        assert pins["sqlalchemy"] < Version("2.1"), pins["sqlalchemy"]
        assert "greenlet" in pins, "greenlet must be pinned alongside sqlalchemy"

    def test_pip_itself_is_pinned(self):
        assert "pip" in _pins()

    def test_no_duplicate_pins(self):
        names = [
            canonicalize_name(l.split("==", 1)[0])
            for l in CONSTRAINTS.read_text().splitlines()
            if l.strip() and not l.startswith("#")
        ]
        dupes = {n for n in names if names.count(n) > 1}
        assert not dupes, dupes


class TestDockerfileHonoursTheLock:
    def test_constraints_copied_and_exported_before_the_first_pip_call(self):
        text = DOCKERFILE.read_text()
        copy_at = text.find("COPY constraints.txt")
        env_at = text.find("ENV PIP_CONSTRAINT=/app/constraints.txt")
        first_pip = re.search(r"^\s*RUN .*pip install", text, re.M)
        assert copy_at != -1 and env_at != -1 and first_pip, "lock wiring missing from Dockerfile"
        assert copy_at < first_pip.start() and env_at < first_pip.start(), (
            "constraints.txt must be copied and PIP_CONSTRAINT set before any pip install"
        )

    def test_lock_is_a_repo_file_not_generated_at_build(self):
        assert CONSTRAINTS.exists() and CONSTRAINTS.stat().st_size > 1000


@pytest.mark.parametrize("pkg", ["fastapi", "pydantic", "torch", "transformers", "alembic"])
def test_load_bearing_packages_are_pinned(pkg):
    assert pkg in _pins()
