Installation
============

Install the base package:

.. code-block:: bash

   pip install put-asunder

Install common development and documentation extras in a local clone:

.. code-block:: bash

   python -m pip install -e ".[dev,graph,viz,docs]"

Notes
-----

- Solver-backed workflows require an available Pyomo-compatible solver.
- Solver support is configured through the local environment rather than a
  dedicated project extra.
- If using Gurobi, configure the ``GRB_LICENSE_FILE`` environment variable.
