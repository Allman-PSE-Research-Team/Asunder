Special Topics
==============

This page documents advanced integration contracts for replacing parts of
Asunder's decomposition loop. Start with
:doc:`../../getting_started/base_decomposition` for a complete configured
workflow before implementing these callables.

Where new code belongs
----------------------

Put reusable algorithms, orchestration, and utilities in ``asunder.base``.
Put fixed-K balance behavior in ``asunder.load_balancing``. Put logic that
depends on NLBNP graph semantics or packaged case studies in
``asunder.nlbnp``. Keep the top-level ``asunder`` namespace as a convenience
facade rather than a second package tree.

Partition-matrix invariant
--------------------------

Initial-column, pricing, and refinement callables exchange binary
co-membership matrices ``Z`` with shape ``(N, N)``. A valid matrix is:

- square and finite;
- symmetric;
- binary with a unit diagonal; and
- transitive, so it represents an equivalence relation.

Entry ``Z[i, j]`` is one exactly when nodes ``i`` and ``j`` share a community.
The decomposition validates custom columns before admitting them to the master
problem. Use
:func:`~asunder.base.utils.graph.validate_partition_matrix` in standalone
extensions and tests.

Initial feasible column generator
---------------------------------

Configure an initial generator through:

.. code-block:: python

   ifc_params = {
       "generator": callable,
       "num": 1,
       "args": {"N": 4},
   }

The decomposition calls the generator with ``**args`` and ``seed=seed``. It
must return a sequence of valid partition matrices; returning ``None`` or an
empty sequence ends the run as infeasible.

This minimal generator returns one all-in-one partition:

.. code-block:: python

   from typing import Any

   import numpy as np

   def initial_columns(
       N: int,
       *,
       seed: int | None = None,
       **kwargs: Any,
   ) -> list[np.ndarray]:
       del seed, kwargs
       return [np.ones((N, N), dtype=int)]

The generator is responsible for satisfying every hard constraint applicable
to initial columns. An all-in-one partition is therefore only a minimal
contract example, not a general feasible generator.

Master problem callable
-----------------------

A master callable has this effective interface:

.. code-block:: python

   from typing import Any

   import numpy as np

   def master_problem(
       A: np.ndarray,
       a: np.ndarray,
       m: float,
       Z_star: list[np.ndarray],
       f_stars: list[float],
       *,
       extract_dual: bool = False,
       **constraints: Any,
   ) -> tuple:
       del A, a, m, constraints
       weights = [1.0] + [0.0] * (len(Z_star) - 1)
       objective = float(f_stars[0])
       if extract_dual:
           return weights, {"mu_dual": 0.0}, objective
       return weights, objective

This implementation only selects the first column and is useful for testing
the callable contract; it is not an optimizing master problem.

Arguments have these meanings:

``A``
   Square adjacency matrix with shape ``(N, N)``.

``a`` and ``m``
   Degree-like vector with shape ``(N,)`` and graph volume scalar.

``Z_star`` and ``f_stars``
   Current partition columns and their objective scores.

``extract_dual``
   False for an integer selection and true for a relaxed solve that supplies
   pricing duals.

When ``extract_dual=False``, return ``(lambda_sol, master_obj_val)``. When it is
true, return ``(lambda_sol, duals, master_obj_val)``. Return ``None`` objective
values to report an infeasible master problem.

Pricing or subproblem callable
------------------------------

A pricing callable receives the graph data and the master's dual dictionary,
then returns ``(sub_obj_val, z_sol)``. ``sub_obj_val`` is the reduced cost and
``z_sol`` is a valid ``(N, N)`` partition matrix.

.. code-block:: python

   from typing import Any

   import numpy as np

   def pricing_problem(
       A: np.ndarray,
       a: np.ndarray,
       m: float,
       duals: dict[str, Any],
       *,
       gamma: float = 1.0,
       seed: int | None = None,
       **kwargs: Any,
   ) -> tuple[float, np.ndarray]:
       del a, m, duals, gamma, seed, kwargs
       # Zero reduced cost tells the loop that no improving column was found.
       return 0.0, np.eye(A.shape[0], dtype=int)

Custom pricing must calculate reduced cost under the same objective and
resolution convention as the master. A positive value above ``tolerance``
causes the column to be added; a non-improving value stops ordinary pricing.

Refinement callable
-------------------

A refinement hook receives the adjacency and a candidate partition. Return a
valid refined matrix or ``None`` when no refinement should be added.

.. code-block:: python

   from typing import Any

   import numpy as np

   from asunder.base.utils import validate_partition_matrix

   def refine_partition(
       A: np.ndarray,
       partition: np.ndarray,
       *,
       seed: int | None = None,
       **kwargs: Any,
   ) -> np.ndarray | None:
       del seed, kwargs
       return validate_partition_matrix(partition, A.shape[0])

The decomposition may call refinement inside the main loop and after it. Keep
state local to one call unless the callable explicitly implements safe restart
semantics. When a function exposes ``shake_rounds``, the decomposition sets it
to zero for in-loop refinement to avoid duplicating the heavier final search.

ModularVFD constraints
----------------------

ModularVFD supports fast component-local constraints, affected-community
predicates, and partition-wide predicates with guided repair. See
:doc:`extending_modular_vfd` for complete examples and the preparation/binding
lifecycle.

Those constraints govern ModularVFD output only. Complete model-wide
enforcement also requires compatible initial-column, master, pricing, warm
start, and final-validation behavior.

NLBNP integration
-----------------

``CorePeripheryPartition`` is an NLBNP-specific structural shortcut, not a
general core-periphery workflow. It merges the detected periphery into one
linear-only group, excludes that group from the original graph, and treats the
remaining core-side connected components as independent communities. Use
``NonlinearBranchAndPrice`` when that structure is insufficient. Detailed graph
labels, worthy edges, attribute derivation, contraction targets, and packaged
case-study schemas are in :doc:`../nlbnp_inputs`.

The packaged ``run_evaluation`` function is application-specific. Prefer the
generic NLBNP or base APIs when the application does not use a packaged
case-study graph schema.

Documentation and validation responsibilities
---------------------------------------------

Public changes should update narrative examples, the matching ``docs/api``
page, and tests of the callable contract. Useful checks are:

.. code-block:: bash

   pytest -m "not legacy"
   ruff check .
   sphinx-build -W --keep-going -b html docs docs/_build/html

Run solver-marked tests when an available solver is configured.
