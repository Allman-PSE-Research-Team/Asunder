Installation
============

Asunder is distributed on PyPI as ``put-asunder`` and supports Python 3.10
through 3.14.

Install and verify
------------------

Install the released package:

.. code-block:: bash

   python -m pip install put-asunder

Verify the import and installed version:

.. code-block:: bash

   python -c "import asunder; print(asunder.__version__)"

The base installation includes NetworkX, NumPy, Pyomo, ``python-igraph``, and
``leidenalg``. Signed Leiden is the default pricing heuristic.

Configure an optimization solver
--------------------------------

``LoadBalancer``, ``NonlinearBranchAndPrice``, and the default reusable
decomposition workflow solve Pyomo master problems. They therefore require an
available Pyomo-compatible solver.

Asunder selects ``gurobi_direct`` by default. Installing ``gurobipy`` does not
provide a Gurobi license; configure a local, WLS, or other supported Gurobi
license before running a solver-backed workflow. A simple availability check
is:

.. code-block:: python

   from asunder import create_solver

   solver = create_solver("gurobi_direct")
   if not solver.available(False):
       raise RuntimeError("Gurobi is not available or is not licensed")

To select a different installed Pyomo solver for the current process, register
it before calling a high-level workflow. For example, after installing
``highspy``:

.. code-block:: bash

   python -m pip install highspy

.. code-block:: python

   from asunder import create_solver
   from asunder.solvers import set_default_solver

   solver = create_solver("appsi_highs")
   if not solver.available(False):
       raise RuntimeError("HiGHS is not available")
   set_default_solver(solver)

The solver executable, license, and solver-specific environment configuration
remain local responsibilities; they are not provided by an Asunder extra.

Optional extras
---------------

``viz``
   Installs Matplotlib and Seaborn for graph, partition, and matrix
   visualization.

   .. code-block:: bash

      python -m pip install "put-asunder[viz]"

``legacy``
   Installs ``cpnet`` for legacy core-periphery heuristics. Current high-level
   workflows do not require it. Upstream compatibility makes this extra
   best-effort on Python 3.13 and 3.14.

   .. code-block:: bash

      python -m pip install "put-asunder[legacy]"

``docs``
   Installs the Sphinx documentation toolchain for a local clone.

   .. code-block:: bash

      python -m pip install -e ".[docs]"

``dev``
   Installs testing, linting, type-checking, build, and contribution tools.

   .. code-block:: bash

      python -m pip install -e ".[dev]"

For a complete contributor environment:

.. code-block:: bash

   python -m pip install -e ".[dev,viz,docs]"

QMETIS availability
-------------------

Released Windows x86-64, Linux x86-64, and macOS universal2 wheels bundle the
QMETIS native library. Pip selects the matching wheel automatically. The
source distribution does not contain native binaries, so ``algorithm="qmetis"``
is unavailable from a plain source install unless a compatible library is
staged as part of a platform-wheel build.

QMETIS is optional at runtime: other pricing algorithms work without its
native library. See :doc:`../reference/qmetis` for supported platforms,
quantization, contraction, and approximation details.

Build the documentation
-----------------------

From a local clone with the ``docs`` extra installed:

.. code-block:: bash

   sphinx-build -W --keep-going -b html docs docs/_build/html

Troubleshooting
---------------

``No executable found`` or ``solver not available``
   Install and configure a Pyomo-compatible solver, then register it with
   ``set_default_solver`` if it is not Gurobi.

Gurobi license errors
   Follow the Gurobi license instructions for your environment. Depending on
   the license type, this may involve ``GRB_LICENSE_FILE`` or WLS credential
   variables.

QMETIS import errors
   Confirm that pip installed a supported platform wheel. Use another pricing
   algorithm when working from an unstaged source distribution.
