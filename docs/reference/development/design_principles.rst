Design Principles
=================

This page summarizes the architectural choices that shape Asunder's package
layout and public APIs.

Layered Package Structure
-------------------------

Asunder is intentionally split into layers:

- ``asunder`` is the public facade
- ``asunder.base`` contains reusable building blocks
- ``asunder.nlbp`` contains the nonlinear branch-and-price application layer

This split exists to keep reusable code reusable. The base layer should remain
general enough to support additional applications without inheriting
application-specific assumptions from the NLBP workflow.

Prefer Reusable Placement
-------------------------

When new functionality is added, the default question should be:

"Could this be reused by another application package?"

If the answer is yes, the code should usually live in ``asunder.base``. If the
answer is no, or if the code is tightly bound to one workflow or one case-study
family, it should live in ``asunder.nlbp`` or in a future peer application
package.

That principle is why:

- branch-and-price infrastructure lives in ``asunder.base.branch_and_price``
- reusable decomposition machinery lives in ``asunder.base.column_generation``
- NLBP-specific refinement logic lives in ``asunder.nlbp.algorithms``
- ``modular_very_fortunate_descent`` remains in ``asunder.base.algorithms``

Keep the Top Level Thin
-----------------------

Top-level ``asunder`` should expose the most useful entry points, but it should
not become a second full package tree. Its job is to provide convenient imports
for orchestration and common workflows, not to duplicate the canonical module
layout.

Graph-First Interfaces
----------------------

The package is designed around graph-derived interfaces:

- adjacency or weight matrices
- partition matrices
- pairwise constraints such as must-link, cannot-link, and worthy edges

This keeps the reusable core compact and allows different application areas to
share the same decomposition machinery even when the original optimization
models differ substantially.

Composable Extension Points
---------------------------

Asunder is meant to be extended by swapping or supplementing a few key pieces:

- initial feasible column generation
- master problem logic
- subproblem or pricing logic
- refinement logic

This is preferable to encoding a large number of hard-wired workflow variants in
the core package. The package should make customization straightforward without
forcing every application into a single monolithic path.

Optional Dependency Boundaries
------------------------------

Some algorithms and workflows rely on optional dependencies such as igraph,
leidenalg, plotting libraries, or solver backends. The package design favors
graceful optionality where possible:

- reusable modules should not pull in heavyweight dependencies unnecessarily
- documentation should make optional requirements visible
- tests should reflect which behavior depends on which extras

Documentation Mirrors the Public Surface
----------------------------------------

The Sphinx API docs are intentionally aligned to the package structure. That
means the docs are part of the architecture, not an afterthought.

When public modules move, the API tree under ``docs/api`` must move with them.
The docs should help users understand both the facade and the canonical module
paths.

Stability Through Explicitness
------------------------------

Asunder favors explicit contracts over hidden conventions. Public callables
should have clear inputs and outputs, typed result containers should describe
what the orchestration layer returns, and application-specific assumptions
should be visible in the module layout rather than hidden inside a generic
namespace.
