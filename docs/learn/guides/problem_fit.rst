Problem Fit and Workflow Choice
===============================

Asunder is useful when a problem has a meaningful undirected graph and the
resulting groups matter operationally or help decompose a larger optimization
problem. This page helps decide whether to use Asunder and which workflow to
start with.

Choose a workflow
-----------------

.. list-table:: Workflow decision guide
   :header-rows: 1
   :widths: 32 32 36

   * - Requirement
     - Recommended entry point
     - Why
   * - A fixed number of balanced or bounded communities
     - ``asunder.load_balancing.LoadBalancer``
     - It includes initial partitions, balance-aware master and pricing logic,
       and VFD refinement.
   * - One NLBNP linear-only separator plus independent communities
     - ``asunder.nlbnp.CorePeripheryPartition``
     - It merges the detected periphery into one linear-only group, then
       excludes it from the original graph to expose connected independent
       core-side communities without column generation.
   * - An NLBNP problem not captured by that structural shortcut
     - ``asunder.nlbnp.NonlinearBranchAndPrice``
     - It enforces the edge rule through column generation. The cardinality rule
       is enforced through column generation only for the exact reformulated case;
       confidence-based and core-periphery modes are structure-specific heuristics.
   * - Custom initial columns, master, pricing, or refinement
     - ``asunder.run_csd_decomposition``
     - It exposes reusable orchestration without imposing one application
       workflow.
   * - Only refinement of an existing partition
     - ``refine_partition_modular_vfd``
     - It improves one candidate and supports application-defined hard
       constraints without running the full decomposition loop.

Follow :doc:`../../getting_started/quickstart` for load balancing,
:doc:`../../getting_started/base_decomposition` for reusable decomposition, or
:doc:`../../getting_started/nlbnp` for the NLBNP entry points.

Good fit
--------

Asunder is usually worth trying when:

- nodes represent real units such as tasks, constraints, assets, scenarios,
  or locations;
- edge weights express meaningful interaction or coupling strength;
- dense local interactions and sparser long-range interactions should influence
  the grouping;
- balanced loads, pairwise grouping rules, or application-defined community
  constraints matter;
- the resulting communities are useful for coordination, interpretation, or a
  downstream optimization workflow; and
- heuristic pricing or refinement is acceptable, even if other stages use an
  exact solver.

Typical graph patterns include shared variables between mathematical
constraints, communication between computing tasks, geographic adjacency,
resource coupling, and interactions across time or scenarios.

Poor fit
--------

Asunder is usually a poor fit when:

- there is no defensible graph representation of the problem;
- the graph is nearly uniform, so it provides little grouping information;
- the partition has no operational or modeling value by itself;
- every stage requires an exact application-specific formulation and no custom
  implementation is planned; or
- important constraints are enforced only during refinement even though other
  column-generation stages can introduce violating partitions.

Check the graph before tuning algorithms
----------------------------------------

Algorithm settings cannot repair an unsuitable graph model. Before comparing
pricing or refinement methods, confirm that:

- node identity is stable and maps back to application data;
- edge meaning and weight scale are documented;
- must-link and cannot-link pairs refer to the intended identifiers;
- node weights describe the quantity that should be balanced; and
- a small known example produces communities that are meaningful to a domain
  expert.

Choose the right constraint mechanism
-------------------------------------

Use built-in ``must_link`` and ``cannot_link`` arguments for pairwise rules.
Use ``K``, ``R``, ``R_bounds``, and node weights for ordinary load balancing.
Use ModularVFD's constraint API when a refinement rule examines community
contents, counts qualifying communities, or compares the whole partition.

Custom refinement constraints do not automatically become master or pricing
constraints. Read
:doc:`../../reference/development/extending_modular_vfd` before treating a
custom rule as an end-to-end model guarantee.

When to customize
-----------------

Customization is appropriate when:

- the initial feasible partitions require domain-specific construction;
- the objective used for pricing differs from the built-in modularity-based
  choices;
- pricing requires a specialized heuristic or exact formulation;
- refinement needs application metadata or non-pairwise hard constraints; or
- the final partition needs domain-specific validation or post-processing.

Put reusable logic in ``asunder.base``. Put workflow-specific behavior in
``asunder.load_balancing``, ``asunder.nlbnp``, or another application package.
The callable contracts are documented in
:doc:`../../reference/development/special_topics`.
