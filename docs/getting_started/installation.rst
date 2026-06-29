Installation
============

Asunder is distributed on PyPI as ``put-asunder``. The base install is enough
for the main package, the load balancing workflow, the default decomposition
APIs, NetworkX-based algorithms, scikit-network algorithms, Pyomo modeling, and
the standard evaluation utilities.

Base Install
------------

Install the released package with:

.. code-block:: bash

   python -m pip install put-asunder

Optional Extras
---------------

Optional extras add dependencies for workflows that are useful but not required
for every installation. Extras can be installed one at a time or combined in a
comma-separated list.

``graph``
   Installs ``python-igraph`` and ``leidenalg``.

   Use this extra when you want the igraph- or leidenalg-backed community
   detection paths, such as calling decomposition routines with
   ``package="igraph"`` or ``package="leidenalg"``. These backends are
   especially useful for larger graph instances where compiled graph routines
   can be faster than pure Python alternatives. ``python-igraph`` is also used
   by some graph automorphism and symmetry-detection helpers.

   .. code-block:: bash

      python -m pip install "put-asunder[graph]"

``viz``
   Installs ``matplotlib`` and ``seaborn``.

   Use this extra when you want plotting helpers under
   ``asunder.base.visualization`` for graph, partition, and matrix inspection.
   It is independent of the graph extra, but commonly installed with it for
   exploratory analysis.

   .. code-block:: bash

      python -m pip install "put-asunder[viz]"

``legacy``
   Installs ``cpnet``.

   Use this extra only when you need legacy core-periphery heuristics that rely
   on ``cpnet``. The core package and current high-level workflows do not
   require it. The legacy extra is best-effort on Python 3.13 and 3.14 because
   it depends on compatibility from the upstream legacy package.

   .. code-block:: bash

      python -m pip install "put-asunder[legacy]"

``docs``
   Installs the Sphinx documentation toolchain: ``sphinx``, ``furo``,
   ``myst-parser``, and ``sphinx-autodoc-typehints``.

   Use this extra from a local clone when you want to build the documentation.

   .. code-block:: bash

      python -m pip install -e ".[docs]"

``dev``
   Installs development tools: ``pytest``, ``pytest-cov``, ``ruff``, ``mypy``,
   and ``pre-commit``.

   Use this extra from a local clone when you want to run tests, linting, type
   checks, or contribution hooks.

   .. code-block:: bash

      python -m pip install -e ".[dev]"

Common Install Recipes
----------------------

Install graph algorithms plus visualization support from PyPI:

.. code-block:: bash

   python -m pip install "put-asunder[graph,viz]"

Install the common contributor environment from a local clone:

.. code-block:: bash

   python -m pip install -e ".[dev,graph,viz,docs]"

Build the documentation after installing the documentation dependencies:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

Python Support
--------------

The core package supports Python 3.10, 3.11, 3.12, 3.13, and 3.14. The
mainstream ``graph`` and ``viz`` extras are also expected to work across those
versions. The ``legacy`` extra is maintained on a best-effort basis on Python
3.13 and 3.14 because it depends on upstream legacy dependencies.

Notes
-----

Solver-backed workflows require an available Pyomo-compatible solver in your
local environment. Asunder includes the Python modeling interfaces needed by
the package, but solver executables, licenses, and solver-specific environment
configuration remain local setup concerns rather than dedicated Asunder extras.

For Gurobi-backed runs, configure your environment as required by Gurobi. In
many setups that includes setting ``GRB_LICENSE_FILE`` before calling
``create_solver("gurobi_direct")`` or another Gurobi solver name.
