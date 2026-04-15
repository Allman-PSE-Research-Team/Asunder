Introduction
============

Asunder is a Python package for constrained network structure detection on undirected
graphs. In practice, that means it helps you take a graph whose nodes and edges are 
subject to constraints and generates a partition that respects those constraints.

The package is built around a simple idea: many parallel computing problems can be
viewed through a graph, and the constrained community structure detection problem can 
be decomposed using a restricted master problem plus a pricing or heuristic subproblem.
Asunder provides reusable tooling for that pattern and leaves room for
application-specific logic where needed.

What Asunder Provides
---------------------

Asunder gives you three things:

- a reusable decomposition layer for graph-based constrained partitioning and
  column-generation style workflows
- a collection of supporting algorithms, utilities, and visualization helpers
- packaged application logic for the current nonlinear branch-and-price
  workflow

The package is therefore useful in two different modes:

- as a reusable toolkit for constrained partitioning and decomposition
- as a concrete application package for the built-in nonlinear branch-and-price
  workflow and its case studies

Package Layout
--------------

The public package is intentionally split into layers.

``asunder``
   The top-level facade. Use this when you want the main convenience entry
   points such as ``run_csd_decomposition``, ``solve_master_problem``,
   ``solve_subproblem``, ``run_evaluation``, or the orchestration/config types.

``asunder.base``
   The reusable layer. This contains algorithms, branch-and-price utilities,
   column-generation modules, evaluation metrics, legacy notebook shims,
   utilities, and visualization helpers.

``asunder.nlbp``
   The application layer for nonlinear branch-and-price workflows. This
   contains case studies, the built-in evaluation runner, and refinement logic
   that is specific to that workflow.

If you are building a new application area, ``asunder.base`` is the starting
point. If you want to use the existing nonlinear branch-and-price workflow,
``asunder.nlbp`` gives you the packaged application-specific pieces.

Mental Model
------------

Most workflows in Asunder follow this rough sequence:

1. Build or derive a graph whose nodes represent constraints, tasks, entities,
   or other units that should be grouped.
2. Encode constraints through available and additional constraints. This could be 
   pairwise structure through adjacency, weights, must-link, cannot-link, or 
   worthy-edge style constraints.
3. Generate one or more initial feasible partition matrices.
4. Run a master problem and a pricing or heuristic subproblem in a loop.
5. Optionally apply a refinement step or application-specific post-processing.
6. Evaluate, inspect, or visualize the resulting partition.

You do not need every layer every time. Some users will call the high-level
top-level API, while others will work directly with ``asunder.base`` modules
and plug in their own master, subproblem, and refinement routines.

When To Start High-Level vs Low-Level
-------------------------------------

Start with the top-level API if:

- you want to run the default decomposition loop quickly
- you are still validating whether your problem is a good fit
- you mainly need orchestration and typed results

Drop into ``asunder.base`` if:

- you need a custom master problem
- you need a custom pricing/subproblem routine
- you want to mix and match reusable utilities and algorithms
- you are building a new application package on top of the reusable layer

Use ``asunder.nlbp`` if:

- you want the built-in nonlinear branch-and-price case studies
- you want the packaged evaluation runner
- you need the NLBP-specific refinement workflow

What Asunder Does Not Do For You
--------------------------------

Asunder does not yet support general-purpose automatic decomposition from a raw
Pyomo model. It expects you to supply either:

- a graph representation of the problem, or
- enough application logic to derive one

It is also not a guarantee that a default heuristic will be appropriate for
every problem. The package is designed so that you can replace initial feasible
column generation, master problem logic, subproblem logic, and refinement logic
when the default pieces are not enough.

Next Steps
----------

- See :doc:`installation` for environment setup.
- See :doc:`quickstart` for a minimal runnable example.
- See :doc:`../learn/guides/problem_fit` for guidance on when the package is a
  good fit and when customization is likely to be necessary.
