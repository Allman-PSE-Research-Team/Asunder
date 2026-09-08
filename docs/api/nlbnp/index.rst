NLBNP API
=========

``asunder.nlbnp.CorePeripheryPartition`` is the NLBNP structural shortcut: a
constrained binary separation identifies a nonlinear/core side and a linear-only
periphery. The periphery becomes one community; excluding it from the original
graph exposes connected independent core-side communities. This is not a
general-purpose core-periphery workflow. Use
``asunder.nlbnp.NonlinearBranchAndPrice`` when that structure is insufficient,
and use ``run_evaluation`` only for the packaged case-study benchmark flow.

.. toctree::
   :maxdepth: 2

   workflow
   algorithms/index
   case_studies/index
