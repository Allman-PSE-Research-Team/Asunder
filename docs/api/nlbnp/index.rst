NLBNP API
=========

``asunder.nlbnp.CorePeripheryPartition`` is the NLBNP structural shortcut: a
constrained binary separation identifies a nonlinear/core side and a linear-only
periphery. The periphery becomes one community; excluding it from the original
graph exposes connected independent core-side communities. This is not a
general-purpose core-periphery workflow. Use
``asunder.nlbnp.NonlinearBranchAndPrice`` when that structure is insufficient,
and use ``run_evaluation`` only for the packaged case-study benchmark flow.
The nonlinear branch-and-price entry point offers exact reformulated (default),
assignment-confidence, and core-periphery cardinality modes; Column generation 
handles the edge-based rule in all 3 modes and the cardinality rule in the 
exactly reformulated mode.
Its general column refiner belongs solely to Stage 1; heuristic cardinality
constraint enforcement is a separate Stage 2 operation on a hard Stage 1 partition.

.. toctree::
   :maxdepth: 2

   workflow
   algorithms/index
   case_studies/index
