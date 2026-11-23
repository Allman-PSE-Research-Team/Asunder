"""Verify native contents and tags of an Asunder platform wheel."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path, PurePosixPath

NATIVE_SUFFIXES = {".dll", ".dylib", ".so"}
VFD_LIBRARY = {
    "linux": "libasunder_vfd.so",
    "macos": "libasunder_vfd.dylib",
    "windows": "asunder_vfd.dll",
}
VFD_SOURCE = Path(__file__).resolve().parents[1] / "native" / "vfd"
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
    """Validate a platform wheel's libraries, QMETIS metadata, VFD license, and tags.

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
    qmetis_libraries = set(asset["libraries"])
    vfd_library = VFD_LIBRARY[platform_key.split("-", 1)[0]]
    expected_libraries = qmetis_libraries | {vfd_library}

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
        if set(bundle["libraries"]) != qmetis_libraries:
            raise RuntimeError(
                "Wheel QMETIS bundle metadata lists unexpected libraries."
            )

        algorithms = PurePosixPath("asunder/load_balancing/algorithms")
        runtime_dir = algorithms / "_native"
        required_runtime_suffixes = {
            algorithms / "_qmetis_wrapper.py",
            algorithms / "_vfd_wrapper.py",
            algorithms / "VFD_cpp.py",
            *(runtime_dir / library for library in expected_libraries),
        }
        wheel_paths = [PurePosixPath(name) for name in names]
        for suffix in required_runtime_suffixes:
            if not any(
                name.parts[-len(suffix.parts) :] == suffix.parts
                for name in wheel_paths
            ):
                raise RuntimeError(f"Wheel is missing required file {suffix}.")

        license_path = runtime_dir / "VFD-LICENSE"
        licenses = [
            name for name in wheel_paths if name.parts[-len(license_path.parts):] == license_path.parts
        ]
        if len(licenses) != 1 or archive.read(str(licenses[0])) != (VFD_SOURCE / "LICENSE").read_bytes():
            raise RuntimeError("Wheel VFD license does not match the source license.")

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
