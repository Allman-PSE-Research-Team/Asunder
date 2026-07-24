"""Validate that a release tag matches the project metadata version."""

from __future__ import annotations

import argparse
from pathlib import Path

import tomllib


def project_version(pyproject: Path) -> str:
    """Read the PEP 621 project version from ``pyproject.toml``."""

    with pyproject.open("rb") as stream:
        return str(tomllib.load(stream)["project"]["version"])


def main() -> int:
    """Verify that a release tag matches the declared project version."""

    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    arguments = parser.parse_args()

    expected = arguments.tag.removeprefix("v")
    actual = project_version(arguments.pyproject)
    if expected != actual:
        raise SystemExit(
            f"Release tag {arguments.tag!r} does not match project version {actual!r}."
        )
    print(f"release_version_ok {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
