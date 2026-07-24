Making Asunder Releases
=======================

This page summarizes the release workflow used for Asunder package builds and
PyPI publication.

Release Checklist
-----------------

1. Update ``pyproject.toml``, ``asunder.__version__``, and
   ``RELEASE_NOTES.md``.
2. If QMETIS changed, update ``_qmetis_assets.json`` with the new immutable
   release tag, ABI, filenames, and SHA-256 digests.
3. Run core, solver, QMETIS native, documentation, and packaging tests.
4. Run the ``Release`` workflow manually without publishing and review all
   three platform wheels plus the source distribution.
5. Optionally run it with ``publish-testpypi`` enabled and install the release
   candidate from TestPyPI on all supported platforms.
6. Review the diff, commit, and create a tag matching the project version,
   such as ``v0.3.0``.
7. Push the tag. GitHub Actions rebuilds and validates every artifact,
   publishes through the protected ``pypi`` environment, attests the
   distributions, and creates the GitHub Release with ``SHA256SUMS``.
8. Install the exact version from PyPI on Windows, Linux, Intel macOS, and
   Apple Silicon macOS.

Publishing Authentication
-------------------------

The PyPI project must trust ``.github/workflows/release.yml`` for the
``pypi`` GitHub environment. TestPyPI uses a separate ``testpypi``
environment. The publish jobs request ``id-token: write`` and use PyPI's
short-lived OIDC credentials; repository Twine passwords and long-lived PyPI
API tokens are not used.

Release Artifacts
-----------------

The workflow publishes one Windows x86-64 wheel, one audited Linux x86-64
wheel, one macOS 11 universal2 wheel, and one source distribution. Each wheel
downloads a pinned QMETIS GitHub Release asset, verifies its checksum, stages
only the QMETIS-named runtime library, installs the completed wheel in a clean
environment, and runs native QMETIS plus end-to-end LoadBalancer smoke tests.
Generic ``metis.dll`` and ``libmetis`` artifacts are explicitly excluded.

The Linux wheel's manylinux tag is produced by ``auditwheel`` from the actual
binary dependencies. Never assign a more permissive manylinux tag manually.
The tag workflow also records GitHub build-provenance attestations and uploads
the exact PyPI distribution set plus checksums to the corresponding GitHub
Release.

Procedure Source
----------------

The automated procedure is maintained in ``.github/workflows/release.yml``.
