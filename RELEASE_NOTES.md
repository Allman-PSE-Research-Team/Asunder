# Release Notes

## v0.1.1 - 2026-02-20

### Added
- Public examples for package usage, including `examples/custom_subproblem.py`.

### Changed
- Updated `README.md` content and project presentation.

### Fixed
- No bug-fix changes in this release.

### Packaging
- Install name remains `put-asunder`.
- Python support remains 3.10, 3.11, 3.12, 3.13.

### Validation
- `pytest`
- `ASUNDER_REQUIRE_SOLVER_TESTS=1 pytest -m solver` (when solver is available)
- `ruff check .`

### Upgrade Impact
- No public API changes expected from `0.1.0` to `0.1.1`.

## v0.1.0 - 2026-02-20

### Added
- Initial `asunder` package from research notebook codebase.
- Column generation orchestration and case studies.
- Core/community evaluation workflows and test suite.

### Changed
- Not applicable (initial release).

### Fixed
- Not applicable (initial release).

### Packaging
- Install name: `put-asunder`.
- Python support: 3.10, 3.11, 3.12, 3.13.

### Validation
- `pytest`
- `ASUNDER_REQUIRE_SOLVER_TESTS=1 pytest -m solver` (when solver is available)
- `ruff check .`
