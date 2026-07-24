import hashlib
import io
import zipfile

import pytest

from tools.fetch_qmetis import (
    GENERATED_FILENAMES,
    asset_url,
    clear_generated_files,
    extract_selected,
    load_manifest,
    verify_digest,
)


def test_pinned_asset_manifest_is_complete():
    manifest = load_manifest()

    assert manifest["release_tag"].startswith("qmetis-v")
    assert manifest["idx_width"] == 64
    assert manifest["real_width"] == 32
    assert {
        "linux-x86_64",
        "macos-arm64",
        "macos-universal2",
        "macos-x86_64",
        "windows-x86_64",
    } == set(manifest["assets"])

    for key, asset in manifest["assets"].items():
        assert key in asset["filename"]
        assert len(asset["sha256"]) == 64
        int(asset["sha256"], 16)
        assert manifest["release_tag"] in asset_url(manifest, asset)
        assert len(asset["libraries"]) == 1
        assert "qmetis" in asset["libraries"][0]
        assert asset["libraries"][0] not in {
            "metis.dll",
            "libmetis.dylib",
            "libmetis.so",
        }


def test_digest_verification_detects_replaced_asset():
    payload = b"qmetis"
    expected = hashlib.sha256(payload).hexdigest()
    verify_digest(payload, expected)

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_digest(payload + b"-changed", expected)


def test_zip_extraction_selects_only_runtime_files(tmp_path):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        root = "qmetis-test-root"
        archive.writestr(f"{root}/lib/qmetis.dll", b"qmetis")
        archive.writestr(f"{root}/lib/metis.dll", b"metis")
        archive.writestr(f"{root}/include/metis.h", b"not packaged")
        archive.writestr(f"{root}/LICENSE", b"license")
        archive.writestr(f"{root}/BUILD-INFO.txt", b"build")

    extract_selected(
        payload.getvalue(),
        "asset.zip",
        ["qmetis.dll"],
        tmp_path,
    )

    assert (tmp_path / "qmetis.dll").read_bytes() == b"qmetis"
    assert not (tmp_path / "metis.dll").exists()
    assert (tmp_path / "QMETIS-LICENSE").read_bytes() == b"license"
    assert not (tmp_path / "metis.h").exists()


def test_generated_file_cleanup_preserves_package_source(tmp_path):
    package_source = tmp_path / "__init__.py"
    package_source.write_text('"""native package"""', encoding="utf-8")
    for filename in GENERATED_FILENAMES:
        (tmp_path / filename).write_text("generated", encoding="utf-8")

    clear_generated_files(tmp_path)

    assert package_source.is_file()
    assert not any((tmp_path / name).exists() for name in GENERATED_FILENAMES)
