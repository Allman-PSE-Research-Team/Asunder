Making Asunder Releases
=======================

This page summarizes the release workflow used for Asunder package builds and
PyPI publication.

Release Checklist
-----------------

1. Activate the project environment and install required extras.
2. Run lint and tests (including solver-marked tests when applicable).
3. Build fresh artifacts in ``dist/`` and validate metadata.
4. Smoke-test installation from the built wheel in a clean environment.
5. Update ``pyproject.toml`` version and ``RELEASE_NOTES.md``.
6. Review git diffs, commit, and create an annotated tag.
7. Upload artifacts with Twine and push tags.
8. Verify installation from the package index in a clean environment.

Procedure Source
----------------

The detailed working checklist is maintained in ``raw/PROCEDURE.md``.
