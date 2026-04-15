# Release Notes

## v0.1.2 - 2026-04-14

### Added
- New heuristic subproblem support in `asunder.base.column_generation.subproblem`.
- New nonlinear branch-and-price refinement module at
  `asunder.nlbp.algorithms.refinement`, centered on
  `refine_partition_linear_group`.
- New generic branch-and-price scaffold under `asunder.base.branch_and_price`,
  including the new symmetry detection implementation in
  `asunder.base.branch_and_price.symmetry_detection`.
- Expanded Sphinx API and narrative documentation for the new package layout.

### Changed
- Reorganized the package into a reusable `asunder.base` namespace and an
  application-specific `asunder.nlbp` namespace.
- Moved reusable modules to `asunder.base`:
  - `asunder.algorithms.*` -> `asunder.base.algorithms.*`
  - `asunder.branch_and_price.*` -> `asunder.base.branch_and_price.*`
  - `asunder.column_generation.*` -> `asunder.base.column_generation.*`
  - `asunder.evaluation.metrics` -> `asunder.base.evaluation.metrics`
  - `asunder.legacy.*` -> `asunder.base.legacy.*`
  - `asunder.utils.*` -> `asunder.base.utils.*`
  - `asunder.visualization.*` -> `asunder.base.visualization.*`
- Moved nonlinear branch-and-price application modules to `asunder.nlbp`:
  - `asunder.case_studies.circle_cutting` -> `asunder.nlbp.case_studies.circle_cutting`
  - `asunder.case_studies.cpcong` -> `asunder.nlbp.case_studies.cpcong`
  - evaluation runner -> `asunder.nlbp.case_studies.runner`
- Kept `modular_very_fortunate_descent` in `asunder.base.algorithms`; it is
  treated as reusable base functionality rather than NLBP-specific logic.
- Retained top-level orchestration and convenience imports in `asunder`,
  including `run_csd_decomposition`, `solve_master_problem`,
  `solve_subproblem`, `CSDDecomposition`, `CSDDecompositionConfig`, and
  `run_evaluation`.

### Fixed
- Updated public docstrings so they render correctly in the Sphinx API docs.
- Corrected installation and narrative documentation to reflect the new
  package structure and supported extras.

### Packaging
- Install name remains `put-asunder`.
- Python support remains 3.10, 3.11, 3.12, 3.13.

### Validation
- `pytest`
- `ASUNDER_REQUIRE_SOLVER_TESTS=1 pytest -m solver` (when solver is available)
- `ruff check .`
- `sphinx-build -b html docs docs/_build/html`

### Upgrade Impact
- This is a breaking namespace release. Existing imports from
  `asunder.algorithms`, `asunder.branch_and_price`, `asunder.column_generation`,
  `asunder.case_studies`, `asunder.evaluation`, `asunder.legacy`,
  `asunder.utils`, and `asunder.visualization` must be updated to the new
  `asunder.base.*` or `asunder.nlbp.*` paths.
- No compatibility shims are provided for the removed top-level package
  namespaces.

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
