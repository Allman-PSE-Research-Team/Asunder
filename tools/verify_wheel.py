"""Verify native contents and tags of an Asunder platform wheel."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path, PurePosixPath

NATIVE_SUFFIXES = {".dll", ".dylib", ".so"}
MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "asunder"
    / "load_balancing"
    / "algorithms"
    / "_qmetis_assets.json"
)


def load_manifest() -> dict:
    """Load the repository's pinned QMETIS asset manifest."""

    with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def verify_wheel(wheel: Path, platform_key: str) -> None:
    """Validate a platform wheel's libraries, provenance, and wheel tags.

    Parameters
    ----------
    wheel
        Wheel archive to inspect.
    platform_key
        Platform entry in the pinned QMETIS asset manifest.

    Raises
    ------
    RuntimeError
        If the archive contains unexpected native libraries, incorrect bundle
        metadata, or universal/Python-ABI-specific tags.
    """

    manifest = load_manifest()
    asset = manifest["assets"][platform_key]
    expected_libraries = set(asset["libraries"])

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        native_names = {
            PurePosixPath(name).name
            for name in names
            if PurePosixPath(name).suffix.lower() in NATIVE_SUFFIXES
        }
        if native_names != expected_libraries:
            raise RuntimeError(
                f"Wheel native libraries {sorted(native_names)} do not equal "
                f"expected {sorted(expected_libraries)}."
            )

        bundle_names = [
            name for name in names if name.endswith("/_qmetis_bundle.json")
        ]
        if len(bundle_names) != 1:
            raise RuntimeError("Wheel must contain exactly one QMETIS bundle manifest.")
        bundle = json.loads(archive.read(bundle_names[0]).decode("utf-8"))
        if bundle["release_tag"] != manifest["release_tag"]:
            raise RuntimeError("Wheel contains the wrong QMETIS release.")
        if bundle["sha256"] != asset["sha256"]:
            raise RuntimeError("Wheel QMETIS checksum metadata is incorrect.")
        if set(bundle["libraries"]) != expected_libraries:
            raise RuntimeError(
                "Wheel QMETIS bundle metadata lists unexpected libraries."
            )

        wheel_metadata_names = [
            name for name in names if name.endswith(".dist-info/WHEEL")
        ]
        if len(wheel_metadata_names) != 1:
            raise RuntimeError("Wheel metadata file was not found.")
        wheel_metadata = archive.read(wheel_metadata_names[0]).decode("utf-8")
        tags = [
            line.removeprefix("Tag: ").strip()
            for line in wheel_metadata.splitlines()
            if line.startswith("Tag: ")
        ]
        if not tags or any("-none-any" in tag for tag in tags):
            raise RuntimeError(f"Native wheel has invalid universal tags: {tags}.")
        if not all(tag.startswith("py3-none-") for tag in tags):
            raise RuntimeError(f"Native wheel is unexpectedly Python-ABI specific: {tags}.")


def main() -> int:
    """Run the platform-wheel verification command-line interface."""

    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--platform", required=True, dest="platform_key")
    arguments = parser.parse_args()
    verify_wheel(arguments.wheel, arguments.platform_key)
    print(f"wheel_verification_ok {arguments.wheel} {arguments.platform_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
