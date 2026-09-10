Introduction
============

Asunder is a Python package for constrained network structure detection on
undirected graphs. Simply put, it divides an undirected graph into communities
under hard constraints. It is designed for problems where the grouping itself
is useful or where grouping makes a larger optimization problem easier to
coordinate or solve.

The basic model
---------------

Every Asunder workflow starts from the same ideas:

``node``
   An item to group, such as a task, mathematical constraint, asset, or
   geographic unit.

``edge``
   A relationship between two nodes. An edge weight can express the strength
   of that relationship.

``community``
   One group of nodes.

``partition``
   A complete assignment of nodes to communities. Most Asunder decomposition
   APIs represent a partition as an ``N x N`` binary matrix: entry ``[i, j]``
   is one when nodes ``i`` and ``j`` share a community.

``hard constraint``
   A rule every returned partition must satisfy. Common examples are
   must-link pairs, cannot-link pairs, and community load bounds.

The graph supplies evidence about which nodes belong together. The constraints
define which otherwise attractive partitions are allowed.

Choose the highest-level workflow that fits
-------------------------------------------

.. list-table:: Workflow guide
   :header-rows: 1
   :widths: 25 41 34

   * - Workflow
     - Use it when
     - Main result
   * - ``LoadBalancer``
     - You need a fixed number of balanced or explicitly bounded communities.
     - A :class:`~asunder.types.DecompositionResult` with a partition matrix
       and label-aware balance metadata.
   * - ``CorePeripheryPartition``
     - An NLBNP constraint graph has one linear-only separator group whose
       exclusion exposes independent communities.
     - A one-dimensional label vector plus NLBNP structural metadata.
   * - ``NonlinearBranchAndPrice``
     - The NLBNP structural shortcut is insufficient and the packaged
       column-generation workflow is needed. Its default exact cardinality
       reformulation uses known worthy edges and nonlinear nodes; confidence
       and core-periphery refinements are alternatives.
     - A :class:`~asunder.types.DecompositionResult` with label-aware metadata.
   * - ``run_csd_decomposition``
     - You need to supply or replace initial columns, the master problem,
       pricing, or refinement.
     - A reusable decomposition result and per-iteration records.

Start with :doc:`quickstart` for ordinary balanced partitioning. Use
:doc:`base_decomposition` only when you need the reusable orchestration layer,
and use :doc:`nlbnp` for the two nonlinear branch-and-price entry points.

What column generation means here
---------------------------------

Some high-level workflows use column generation internally. You do not need to
understand it to call ``LoadBalancer``. ``CorePeripheryPartition`` bypasses
column generation and uses the NLBNP structural shortcut described in
:doc:`nlbnp`.

At a conceptual level:

1. An initial-column generator proposes one or more feasible partitions.
2. A master problem scores and combines the available candidates.
3. A pricing method searches for another useful candidate.
4. An optional refinement heuristic improves candidate partitions.
5. The process stops when no useful candidate is found or a configured limit
   is reached.

Users of :doc:`base_decomposition` can replace those pieces. Detailed callable
contracts live in :doc:`../reference/development/special_topics`.

Package organization
--------------------

``asunder``
   Convenience imports for reusable decomposition, solvers, configuration,
   and result types.

``asunder.load_balancing``
   The complete balanced or bounded graph-partitioning workflow.

``asunder.nlbnp``
   The NLBNP linear-only-separator shortcut and nonlinear branch-and-price
   workflow.

``asunder.base``
   Reusable algorithms, column-generation components, utilities, evaluation,
   and visualization. Most users only need this layer when customizing a
   workflow.

What Asunder does not infer
---------------------------

Asunder does not automatically turn a raw optimization model into a graph. You
must supply a NetworkX graph, an adjacency matrix, or application logic that
constructs one. The quality of the result depends on whether that graph and its
constraints capture relationships that matter in the application.

It also does not guarantee that a generic heuristic will solve every custom
constraint effectively. ModularVFD supports extensible community and
partition constraints, but difficult rules may need specialized repair logic
and enforcement in other column-generation stages.

Next steps
----------

- :doc:`installation` explains package, solver, and optional-feature setup.
- :doc:`quickstart` builds a first balanced partition.
- :doc:`../learn/guides/problem_fit` helps choose or reject a workflow.
- :doc:`../reference/development/extending_modular_vfd` explains custom
  refinement constraints.
