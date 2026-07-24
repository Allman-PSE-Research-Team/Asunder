"""Fetch and stage a pinned QMETIS release asset for wheel construction."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT
    / "asunder"
    / "load_balancing"
    / "algorithms"
    / "_qmetis_assets.json"
)
DEFAULT_DESTINATION = (
    PROJECT_ROOT
    / "asunder"
    / "load_balancing"
    / "algorithms"
    / "_native"
)
QMETIS_GENERATED_FILENAMES = {
    "libqmetis.so",
    "libqmetis.dylib",
    "qmetis.dll",
    "QMETIS-LICENSE",
    "QMETIS-BUILD-INFO.txt",
    "_qmetis_bundle.json",
}
CONFLICTING_METIS_FILENAMES = {
    "libmetis.so",
    "libmetis.dylib",
    "metis.dll",
}
# Remove leftovers from builds made before Asunder switched to shipping only
# QMETIS-named runtimes. These files are cleanup targets, never wheel inputs.
GENERATED_FILENAMES = (
    QMETIS_GENERATED_FILENAMES | CONFLICTING_METIS_FILENAMES
)


def detect_platform() -> str:
    """Return the manifest asset key for the current host platform.

    Returns
    -------
    str
        A normalized operating-system and architecture key.

    Raises
    ------
    RuntimeError
        If the current platform has no pinned QMETIS asset.
    """

    machine = platform.machine().lower().replace("-", "_")
    if machine in {"amd64", "x64"}:
        machine = "x86_64"
    elif machine in {"aarch64", "arm64"}:
        machine = "arm64"

    if sys.platform == "linux" and machine == "x86_64":
        return "linux-x86_64"
    if sys.platform == "win32" and machine == "x86_64":
        return "windows-x86_64"
    if sys.platform == "darwin" and machine in {"x86_64", "arm64"}:
        return f"macos-{machine}"
    raise RuntimeError(
        f"No pinned QMETIS asset supports {sys.platform!r}/{platform.machine()!r}."
    )


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Load and minimally validate a QMETIS asset manifest.

    Parameters
    ----------
    path
        JSON manifest to load.

    Returns
    -------
    dict
        Parsed manifest data.

    Raises
    ------
    ValueError
        If a required top-level field is absent.
    """
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    required = {"release_tag", "repository", "idx_width", "real_width", "assets"}
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"QMETIS asset manifest is missing {sorted(missing)}.")
    return manifest


def asset_url(manifest: dict, asset: dict) -> str:
    """Construct the immutable GitHub Release URL for an asset."""

    return (
        f"https://github.com/{manifest['repository']}/releases/download/"
        f"{manifest['release_tag']}/{asset['filename']}"
    )


def download(url: str) -> bytes:
    """Download an asset and return its raw bytes."""

    request = Request(url, headers={"User-Agent": "Asunder-wheel-builder"})
    with urlopen(request, timeout=120) as response:
        return response.read()


def verify_digest(payload: bytes, expected: str) -> None:
    """Verify a payload against an expected SHA-256 hexadecimal digest.

    Raises
    ------
    RuntimeError
        If the computed digest differs from ``expected``.
    """

    actual = hashlib.sha256(payload).hexdigest()
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"QMETIS asset checksum mismatch: expected {expected}, got {actual}."
        )


def _find_member(
    members: list[str],
    suffix: PurePosixPath,
) -> str:
    """Find the unique archive member whose path ends with ``suffix``."""

    matches = [
        member
        for member in members
        if PurePosixPath(member).parts[-len(suffix.parts) :] == suffix.parts
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one archive member ending in {suffix}, found {matches}."
        )
    return matches[0]


def _copy_stream(source: BinaryIO, destination: Path) -> None:
    """Copy a binary stream to ``destination``."""

    with destination.open("wb") as output:
        shutil.copyfileobj(source, output)


def extract_selected(
    payload: bytes,
    filename: str,
    libraries: list[str],
    destination: Path,
) -> None:
    """Extract only selected QMETIS runtime and provenance files.

    Parameters
    ----------
    payload
        Bytes of a ZIP or gzipped tar archive.
    filename
        Asset filename, used to select the archive reader.
    libraries
        QMETIS-named runtime files expected under the archive's ``lib``
        directory.
    destination
        Existing directory into which the selected files are copied.
    """

    wanted = {library: PurePosixPath("lib", library) for library in libraries}
    wanted["QMETIS-LICENSE"] = PurePosixPath("LICENSE")
    wanted["QMETIS-BUILD-INFO.txt"] = PurePosixPath("BUILD-INFO.txt")

    if filename.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            for output_name, suffix in wanted.items():
                member = _find_member(names, suffix)
                with archive.open(member, "r") as source:
                    _copy_stream(source, destination / output_name)
    elif filename.endswith(".tar.gz"):
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            regular_files = [
                member.name for member in archive.getmembers() if member.isfile()
            ]
            for output_name, suffix in wanted.items():
                member_name = _find_member(regular_files, suffix)
                member = archive.getmember(member_name)
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Could not read archive member {member_name}.")
                with source:
                    _copy_stream(source, destination / output_name)
    else:
        raise ValueError(f"Unsupported QMETIS archive format: {filename}")


def clear_generated_files(destination: Path) -> None:
    """Remove previously staged QMETIS files from ``destination``.

    Generic METIS filenames from older builds are also removed so they cannot
    accidentally enter a new platform wheel.
    """

    destination.mkdir(parents=True, exist_ok=True)
    for filename in GENERATED_FILENAMES:
        path = destination / filename
        if path.is_file():
            path.unlink()


def stage_asset(
    platform_key: str,
    destination: Path = DEFAULT_DESTINATION,
    *,
    manifest_path: Path = MANIFEST_PATH,
) -> dict:
    """Download, verify, and stage one platform's pinned QMETIS asset.

    Parameters
    ----------
    platform_key
        Key in the manifest's ``assets`` mapping.
    destination
        Package directory that receives the native runtime and provenance
        files.
    manifest_path
        Asset manifest used as the source of release metadata and checksums.

    Returns
    -------
    dict
        Bundle metadata written alongside the staged runtime.
    """

    manifest = load_manifest(manifest_path)
    try:
        asset = manifest["assets"][platform_key]
    except KeyError as exc:
        supported = ", ".join(sorted(manifest["assets"]))
        raise ValueError(
            f"Unsupported QMETIS platform {platform_key!r}; expected one of {supported}."
        ) from exc

    url = asset_url(manifest, asset)
    payload = download(url)
    verify_digest(payload, asset["sha256"])

    clear_generated_files(destination)
    extract_selected(
        payload,
        asset["filename"],
        list(asset["libraries"]),
        destination,
    )

    bundle = {
        "release_tag": manifest["release_tag"],
        "library_version": manifest.get("library_version"),
        "platform": platform_key,
        "idx_width": manifest["idx_width"],
        "real_width": manifest["real_width"],
        "asset": asset["filename"],
        "sha256": asset["sha256"],
        "source_url": url,
        "libraries": list(asset["libraries"]),
    }
    metadata_path = destination / "_qmetis_bundle.json"
    metadata_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle


def main() -> int:
    """Run the QMETIS asset-staging command-line interface."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform",
        dest="platform_key",
        help="Pinned asset key; defaults to the running platform.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="Directory in which to stage runtime libraries.",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    arguments = parser.parse_args()

    platform_key = arguments.platform_key or detect_platform()
    bundle = stage_asset(
        platform_key,
        arguments.destination.resolve(),
        manifest_path=arguments.manifest.resolve(),
    )
    print(
        f"Staged {bundle['release_tag']} for {bundle['platform']} "
        f"from {bundle['asset']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
