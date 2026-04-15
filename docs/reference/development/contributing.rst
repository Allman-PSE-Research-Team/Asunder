Contributing to Asunder
=======================

This page summarizes the expected workflow for code, documentation, and test
contributions.

Development Setup
-----------------

For a local clone, install the development, documentation, and common optional
extras:

.. code-block:: bash

   python -m pip install -e ".[dev,docs,graph,viz]"

Solver-backed workflows require a working Pyomo-compatible solver in the local
environment. Solver support is not provided through a dedicated project extra;
configure the solver separately.

Common Commands
---------------

Run the default test suite:

.. code-block:: bash

   pytest -m "not legacy"

Run solver-marked tests when a solver is available:

.. code-block:: bash

   ASUNDER_REQUIRE_SOLVER_TESTS=1 pytest -m solver

Build the documentation locally:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

Run linting:

.. code-block:: bash

   ruff check .

Contribution Expectations
-------------------------

Contributions should preserve the current package layering:

- reusable code belongs in ``asunder.base``
- nonlinear branch-and-price specific code belongs in ``asunder.nlbp``
- top-level ``asunder`` should remain a thin facade for orchestration and
  convenience entry points

When deciding where new code belongs, prefer the most reusable placement that
does not force application-specific logic into the base layer.

In practice, that means:

- put reusable algorithms, utilities, and decomposition helpers in
  ``asunder.base``
- put NLBP-specific refinement and case-study logic in ``asunder.nlbp``
- avoid importing ``asunder.nlbp`` from reusable modules unless that dependency
  is genuinely intended

Tests and Documentation
-----------------------

Every public-facing change should be reflected in tests and docs.

That is especially important for:

- namespace moves
- new public functions or classes
- changes to orchestration behavior
- changes to case-study expectations

The API docs are maintained explicitly through hand-authored ``.rst`` pages, so
moving a module is not enough on its own. If you add, rename, or move a public
module, update the Sphinx tree under ``docs/api`` as part of the same change.

Suggested Pull Request Checklist
--------------------------------

Before opening or merging a change, check the following:

- the code lives in the correct layer: ``base`` vs ``nlbp``
- tests covering the changed public behavior were added or updated
- docs were updated where the public surface changed
- ``pytest`` passes for the affected scope
- the Sphinx build succeeds if documentation was touched

Review Style
------------

Good contributions tend to be small, well-scoped, and explicit about intent.
When a change introduces a new extension point, a new application-specific
workflow, or a new dependency boundary, explain that design choice in the code
and in the accompanying documentation.
