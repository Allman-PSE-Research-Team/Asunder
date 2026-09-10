Top-Level API
=============

Choose an entry point by task before consulting the full signatures below.

.. list-table:: Goal-to-entry-point map
   :header-rows: 1
   :widths: 38 32 30

   * - Goal
     - Entry point
     - Namespace
   * - Balanced or bounded graph communities
     - ``LoadBalancer``
     - ``asunder.load_balancing``
   * - NLBNP constraints enforced using core-periphery detection
     - ``CorePeripheryPartition``
     - ``asunder.nlbnp``
   * - NLBNP constraints enforced using community detection
     - ``NonlinearBranchAndPrice``
     - ``asunder.nlbnp``
   * - Reusable configured decomposition
     - ``run_csd_decomposition``
     - ``asunder``
   * - Reusable object for repeated decomposition runs
     - ``CSDDecomposition``
     - ``asunder``
   * - Refine one partition with custom constraints
     - ``refine_partition_modular_vfd``
     - ``asunder``
   * - Configure a process-wide solver
     - ``create_solver`` and ``set_default_solver``
     - ``asunder`` and ``asunder.solvers``
   * - Run packaged NLBNP case-study evaluation
     - ``run_evaluation``
     - ``asunder.nlbnp``

See :doc:`../getting_started/index` for complete examples. The API reference
below documents exact signatures and return types.

Package entry points
--------------------

Asunder package
^^^^^^^^^^^^^^^

.. automodule:: asunder

The top-level package re-exports common reusable entry points. Canonical API
documentation remains on the module pages to avoid duplicate object targets.

Common package-level imports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``asunder.run_csd_decomposition``
- ``asunder.solve_master_problem``
- ``asunder.solve_subproblem``
- ``asunder.run_evaluation``
- ``asunder.create_solver``
- ``asunder.refine_partition_modular_vfd``
- ``asunder.CSDDecomposition``
- ``asunder.CSDDecompositionConfig``
- ``asunder.IterationRecord``
- ``asunder.DecompositionResult``

Application package imports
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``asunder.load_balancing.LoadBalancer``
- ``asunder.nlbnp.CorePeripheryPartition``
- ``asunder.nlbnp.NonlinearBranchAndPrice``
- ``asunder.nlbnp.run_evaluation``

Configuration
-------------

.. automodule:: asunder.config
   :members:
   :member-order: bysource
   :show-inheritance:

Orchestrator
------------

.. automodule:: asunder.orchestrator
   :members:
   :member-order: bysource
   :show-inheritance:

Solvers
-------

.. automodule:: asunder.solvers
   :members:
   :member-order: bysource
   :show-inheritance:

Result types
------------

.. automodule:: asunder.types
   :members:
   :member-order: bysource
   :show-inheritance:
