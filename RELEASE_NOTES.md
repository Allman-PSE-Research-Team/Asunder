# Release Notes

## v0.4.0 - 2026-09-24

### Added

- ModularVFD now supports custom component-local, community-wide, and
  partition-wide constraints, including quantified predicates, original-node
  provenance, and guided repair. These constraints apply to ModularVFD
  refinement; other column-generation stages require their own enforcement.
- `NonlinearBranchAndPrice` now offers reformulated, assignment-confidence,
  and core-periphery cardinality modes. The reformulated mode is the default:
  it derives the maximum feasible linear-only set and contracts its exact
  must-link components.
- Column generation can retain CSR adjacency and mixed dense/CSR column pools.
  New columns use Boolean dense or CSR storage according to the storage policy
  and measured density. Incompatible dense operations have a configurable
  estimated working-set cap, 512 MiB by default.
- An opt-in `persistent_master` mode incrementally updates one Gurobi
  restricted master for the built-in base and load-balancing workflows.
- Added a pairwise feasibility projection and DSATUR-based initial columns for
  constrained partitions.
- Core-periphery detection now supports contracted or original-graph targets,
  named fit diagnostics, and optional rank-2 spectral embedding.

### Changed

- Signed Leiden is now the default pricing heuristic. Top-level `resolution`
  applies across pricing, scoring, and QMETIS; backends that cannot honor a
  non-default value reject it.
- Load balancing uses an independent, weight-aware LB VFD again, with modularity
  resolution support. Explicit `R_bounds` are the primary load constraint;
  `K` and `R` derive convenient bounds. LB/VFD node loads require positive
  integers, while generic CSD can forward real weights to custom hooks.
- CSD owns shared `must_link`, `cannot_link`, and `node_weights` inputs. It
  resolves and forwards them through contraction and compatible hooks;
  duplicating standard pairwise constraints in hook kwargs now raises an error.
- VFD replaces `max_K_increase` with symmetric `K_search_radius` and removes
  `very_fortunate_descent_legacy`. ModularVFD uses `candidate_Ks` for explicit
  output-count searches and `clustering_Ks` for co-association construction.
- NLBNP requires at least one worthy edge present in the input graph. Stage 1
  refinement is configured separately from cardinality refinement, and the
  ambiguous top-level `refine` switch is removed. `must_group` identifies the
  nonlinear detection block without merging independent original-graph
  communities.
- The public `extract_dual` decomposition option is removed: relaxed master
  solves provide pricing duals, while final integer solves do not. Iteration
  records now expose read-only column and score prefix views instead of
  copying growing lists at every iteration.
- QMETIS now uses an Asunder-specific native binding and bundled
  `qmetis-v5.2.1-modularity.3` assets; the external `metis` Python dependency
  is removed. The `graph` extra is also removed, with `python-igraph` and
  `leidenalg` required by the base package.
- The README and narrative documentation now lead with complete workflow
  examples; backend, sparse-storage, load-unit, and constraint-extension
  details are collected in reference guides.

### Fixed

- Enforced unworthy-edge must-links, pairwise feasibility, and contracted
  warm-start consistency through CSD refinement and final selection.
- Corrected sparse-dual aggregation, reduced-cost reporting, and one-level
  ModifiedLouvain resolution handling in pricing.
- Preserved supplied dense/CSR column representations, including mixed pools;
  bounded dense scratch, contraction, and expansion without forcing CSR after
  a dense-cap failure.
- Corrected NLBNP core-periphery roles: the detected periphery becomes the
  linear-only group, and independent core-side communities are recovered from
  the original graph. Nonlinear detection blocks remain intact.
- Single-component contraction now reports `feasibility_unchecked` when custom
  master, pricing, and refinement hooks were skipped, rather than claiming
  those custom constraints were checked.
- Base, load-balancing, and persistent master solves now reject non-optimal
  termination instead of using potentially invalid values or duals;
  infeasibility still returns the existing `None` result.

## v0.3.0 - 2026-07-24

### Added
- Added QMETIS as a load-balancing-specific pricing heuristic,
  selectable with `LoadBalancer(..., algorithm="qmetis")`.
- Added overflow-aware integer quantization for fractional dual-adjusted edge
  weights and exact float64 reduced-cost rescoring of QMETIS candidates.
- Added an explicit `QMETISApproximationWarning` when unsupported diagonal
  weights, including contraction-generated internal-edge mass, are omitted
  from QMETIS candidate generation; exact rescoring retains those values.
- Added pinned, checksum-verified QMETIS release assets for Windows x86-64,
  Linux x86-64, and macOS Intel, ARM64, and universal2 builds.
- Added platform-wheel validation, native smoke tests, TestPyPI release
  candidates, and PyPI Trusted Publishing automation.
- Added shared pricing-dual and load-balance-bound utilities plus generic
  `subproblem_params` wiring for application-specific pricing adapters.
- Added `node2comp` result metadata for contracted decompositions so
  component-level iteration columns remain interpretable.

### Changed
- Platform wheels now bundle QMETIS `qmetis-v5.2.1-modularity.1` with the
  `idx64-real32` ABI, while source distributions contain no native libraries.
- Wheels include only the QMETIS-named runtime (`qmetis.dll`,
  `libqmetis.so`, or `libqmetis.dylib`), avoiding a conflicting generic
  METIS library.
- Pricing call dispatch now filters arguments from the callable signature
  instead of relying on function names.
- Removed the unused duplicate `contract_adj_matrix_cp` helper; active
  core-periphery workflows enforce grouping blocks directly.

### Fixed

- Improved must-link graph contraction by remapping cannot-links and initial-column constraints to contracted components, validating and rescoring warm starts, and rejecting contradictory constraints.
- Preserved unworthy-edge and nonlinear-node grouping through final core-periphery component splitting across SPEC, GA, and KL workflows.

## v0.2.6 - 2026-07-03

### Added
- Added `project_partition_ilp`, a load-balancing feasibility projection that
  maps a fractional or starting partition to the nearest feasible exact-`K`
  load-balanced partition under must-link, cannot-link, and size constraints.
- Added `projection_repair` and `projection_time_limit` to `LoadBalancer`.
  Projection repair runs only after the final post-loop VFD refinement fails,
  with a default 15-second best-effort solver time limit.
- Added regression coverage for projection feasibility, fixed-`K` rejection,
  and solver-parameter usage in solver-backed tests.

### Changed
- Documented feasibility projection as a general custom-refinement pattern for
  constrained partitioning, with load balancing as the current concrete
  implementation.
- Kept VFD as the normal load-balancing refinement path while using projection
  only as a final feasible repair option.

## v0.2.5 - 2026-07-03

### Changed:
- Implemented a two-tier incumbent policy for the VFD algorithm
  - `modular_VFD.py`: added shared unnormalized reference scoring helper.
  - `modular_VFD.py` and `VFD.py`: track `best_improving` and `best_feasible` separately, returning the improving incumbent first and the feasible fallback otherwise.
  - `test_review_edge_cases.py`: added regression coverage for an unattainably good reference partition.
### Fixed
- Fixed `leidenalg` `signed_leiden`'s metric computation bug.

## v0.2.4 - 2026-06-28

### Changed
- Expanded README and installation documentation for optional extras, including
  `graph`, `viz`, `legacy`, `docs`, and `dev`, with clearer installation
  procedures, usage guidance, Python support notes, and solver setup notes.

### Fixed
- Fixed `leidenalg` metric extraction so each supported Leiden backend reports
  an algorithm-appropriate quality value, including signed and CPM/surprise
  variants.

## v0.2.3 - 2026-06-26

### Added
- Added `leidenalg`-backed heuristic subproblem support for:
  - `leiden`
  - `signed_leiden`
  - `cpm_leiden`
  - `surprise_leiden`
  - `signed_surprise_leiden`
- Added `igraph` support for `cpm_leiden`.
- Added `RCCS` routing as a custom heuristic subproblem in the NLBNP and load-balancing workflows.

### Changed
- Updated `heuristic_subproblem` routing so signed and CPM algorithms use the appropriate signed/negative-weight-aware backend path.
- Removed in-subproblem refinement from modularity, LPA, and custom heuristic subproblem calls.
- Standardized active custom heuristic routing to `spectral`, `full_louvain`, and `RCCS`.
- Updated case-study evaluation routing so package-backed algorithms are selected from `networkx`, `igraph`, or `leidenalg` consistently.
- Updated public docstrings for `CSD_decomposition`, `CSDDecompositionConfig`, `LoadBalancer`, and `run_nonlinear_branch_and_price` with the current algorithm/package matrix.

### Fixed
- Fixed load-balancing decomposition so custom heuristic algorithms use `custom_heuristic_subproblem` instead of being incorrectly routed through package-backed `heuristic_subproblem`.
- Removed stale `refine_in_subproblem` configuration and call-site usage.
- Corrected several docstring typos and algorithm descriptions, including RCCS community wording and partition spelling.

## v0.2.2 - 2026-06-19

### Added
- Added Python 3.14 support for the core package and mainstream extras.
- Added `refine_post_loop` controls to the decomposition, load-balancing, and NLBNP workflows.
- Added `max_iterations` to the high-level load-balancing workflow.

## v0.2.1 - 2026-06-05

### Added
- Added `asunder.load_balancing.LoadBalancer`, a high-level load-balanced graph partitioning workflow.
- Added `asunder.nlbnp.NonlinearBranchAndPrice` for generic nonlinear branch-and-price problems.
- Added `asunder.nlbnp.CorePeripheryPartition` for the NLBNP structural
  shortcut that separates one linear-only group from independent components.
- Added core-periphery detection and NLBNP refinement utilities.
- Added regression and integration coverage for load-balancing, decomposition, constraint, and large sparse-graph edge cases.
- Added `__version__` attribute.

### Changed
- Renamed the `asunder.nlbp` package and NLBP references to `asunder.nlbnp` and NLBNP.
- Replaced `defaultdict` constraint handling with regular dictionaries and safe internal normalization.
- Improved node-label mapping, constraint validation, infeasibility handling, and large sparse-graph partition generation.
- Expanded README and documentation coverage for load balancing and NLBNP workflows.
- Updated title underline in docs.

### Removed
- Removed the old `asunder.nlbp` package path.

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
  treated as reusable base functionality rather than nlbp-specific logic.
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
