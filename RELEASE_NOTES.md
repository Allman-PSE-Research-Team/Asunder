# Release Notes

## v0.1.0 - Initial package release

## Scope
- Initial `asunder` package from research notebook codebase. 
- Column generation orchestration + case studies.
- Core/community evaluation workflows and tests.

## Packaging
- Install name: `put-asunder`
- Python support: 3.10, 3.11, 3.12, 3.13

## Testing Notes
- pytest
- ASUNDER_REQUIRE_SOLVER_TESTS=1 pytest -m solver (when solver is available)
- ruff check .
