"""Verify that an Asunder source distribution is native-library free."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path, PurePosixPath

NATIVE_SUFFIXES = {".dll", ".dylib", ".so"}


def verify_sdist(archive_path: Path) -> None:
    """Verify that an sdist has build inputs but no native libraries.

    Parameters
    ----------
    archive_path
        Gzipped source-distribution archive to inspect.

    Raises
    ------
    RuntimeError
        If native code is present or a required QMETIS build input is absent.
    """

    with tarfile.open(archive_path, mode="r:gz") as archive:
        names = [PurePosixPath(member.name) for member in archive.getmembers()]

    native_files = [
        str(name) for name in names if name.suffix.lower() in NATIVE_SUFFIXES
    ]
    if native_files:
        raise RuntimeError(
            f"Source distribution contains native libraries: {native_files}."
        )

    required_suffixes = {
        PurePosixPath(
            "asunder/load_balancing/algorithms/_qmetis_assets.json"
        ),
        PurePosixPath("tools/fetch_qmetis.py"),
    }
    for suffix in required_suffixes:
        if not any(name.parts[-len(suffix.parts) :] == suffix.parts for name in names):
            raise RuntimeError(
                f"Source distribution is missing required file {suffix}."
            )


def main() -> int:
    """Run the source-distribution verification command-line interface."""

    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    arguments = parser.parse_args()
    verify_sdist(arguments.archive)
    print(f"sdist_verification_ok {arguments.archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
