Special Topics
==============

This page collects advanced integration notes for extending Asunder beyond the
default workflows.

Choosing Where New Code Belongs
-------------------------------

Before adding new logic, decide whether it is reusable or application-specific.

Put code in ``asunder.base`` when:

- it can support multiple application packages
- it does not depend on application-specific assumptions
- it belongs to reusable decomposition, algorithm, utility, or visualization
  infrastructure

Put code in ``asunder.nlbp`` when:

- it is specific to the nonlinear branch-and-price workflow
- it depends on NLBP case-study conventions
- it exists mainly to support the built-in evaluation path

Initial Feasible Column Generator Contract
------------------------------------------

The decomposition loop expects an initial feasible column generator through
``ifc_params`` unless an existing column pool is supplied.

The expected ``ifc_params`` shape is:

.. code-block:: python

   {
       "generator": callable,
       "num": int,
       "args": {...},
   }

The generator should return a list of partition matrices, each typically shaped
``(N, N)``.

Master Problem Callable
-----------------------

Custom master callables must accept:

- ``A, a, m, Z_star, f_stars``
- keyword constraints such as ``must_link``, ``cannot_link``, and other
  supported additional constraints

and return either:

- ``(lambda_sol, master_obj_val)`` for integer master mode, or
- ``(lambda_sol, duals, master_obj_val)`` when dual extraction is enabled

In other words, the master callable is responsible for honoring the current
column pool and for returning either an integer selection or a relaxed solution
plus dual information.

Subproblem Callable
-------------------

Custom subproblem callables must accept:

- ``A, a, m, duals``

and return:

- ``(sub_obj_val, z_sol)``

where ``z_sol`` is a partition or co-association matrix compatible with the
rest of the decomposition loop.

Refinement Callable
-------------------

Refinement hooks are passed through ``refine_params`` and are expected to look
like a callable of the form:

.. code-block:: python

   refine_func(A=A, partition=z_or_wz, **kwargs)

The callable should return either:

- a refined partition matrix, or
- ``None`` when no refined result should be added

Use refinement in ``asunder.base`` only when the logic is reusable. Keep
workflow-specific refinement in application packages such as ``asunder.nlbp``.

Case-Study Evaluation Contract
------------------------------

The built-in ``run_evaluation`` path is application-specific and lives in
``asunder.nlbp.case_studies.runner``. It assumes a packaged case-study style
graph schema rather than a completely generic graph input.

If you only need reusable decomposition behavior, prefer working directly with
the base-layer APIs rather than routing through ``run_evaluation``.

Documentation Responsibilities
------------------------------

Because the API docs are hand-authored, changes to public modules should be
paired with updates to:

- the relevant ``docs/api/...`` pages
- any examples that import the moved or renamed modules
- narrative docs if the public mental model changed

Useful Local Validation Commands
--------------------------------

For changes that touch public APIs or docs, the most useful checks are:

.. code-block:: bash

   pytest -m "not legacy"
   sphinx-build -b html docs docs/_build/html

If solver-backed workflows were affected and a solver is available locally, also
run the solver-marked tests.
