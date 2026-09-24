# Asunder

![Asunder package banner](assets/asunder.gif)

Asunder is a Python package for constrained network structure detection
(constrained graph clustering) on undirected graphs. In other words, it
partitions an undirected graph while respecting hard grouping rules. A
node can represent a task, mathematical constraint, asset, location, or other
item; an edge records a relationship between two nodes; and the result assigns
every node to a community.

For example, Asunder can keep specified nodes together, prevent other nodes
from sharing a community, and keep community sizes or workloads within chosen
bounds. It also provides reusable column-generation tools for applications
that need custom initial partitions, master problems, pricing, or refinement.

Development is led by [Andrew Allman's Process Systems Research
Team](https://allmanaa.engin.umich.edu/) at the University of Michigan.

## Install

Asunder supports Python 3.10 through 3.14 and is distributed on PyPI as
`put-asunder`:

```bash
python -m pip install put-asunder
python -c "import asunder; print(asunder.__version__)"
```

The base installation includes NetworkX, NumPy, Pyomo, `python-igraph`, and
`leidenalg`. Signed Leiden is the default pricing heuristic.

`LoadBalancer`, `NonlinearBranchAndPrice`, and the default reusable
decomposition workflow require an available Pyomo-compatible optimization
solver. Asunder selects Gurobi by default. Installing `gurobipy` does **not**
provide a Gurobi license, so configure a working local, WLS, or other supported
license before running those workflows. See the [installation
guide](https://asunder.readthedocs.io/en/latest/getting_started/installation.html)
for a solver check and alternative-solver setup.

Optional extras are available for visualization and legacy core-periphery
heuristics:

```bash
python -m pip install "put-asunder[viz]"
python -m pip install "put-asunder[legacy]"
```

The `legacy` extra is best-effort on Python 3.13 and 3.14. Current high-level
workflows do not require it.

## Choose a workflow

| Goal | Start with | Solver required? |
| --- | --- | --- |
| Create a fixed number of balanced or explicitly bounded communities | `asunder.load_balancing.LoadBalancer` | Yes |
| Use the NLBNP structural shortcut to isolate one linear-only group and recover independent communities | `asunder.nlbnp.CorePeripheryPartition` | No |
| Run the packaged nonlinear branch-and-price workflow | `asunder.nlbnp.NonlinearBranchAndPrice` | Yes |
| Replace initial columns, master logic, pricing, or refinement | `asunder.run_csd_decomposition` | Yes with the default master |
| Refine an existing partition with extensible hard constraints | `asunder.refine_partition_modular_vfd` | No |

If you are unsure, start with the [problem-fit and workflow-choice
guide](https://asunder.readthedocs.io/en/latest/learn/guides/problem_fit.html).

## Load-balancing quickstart

`LoadBalancer` is the most direct workflow when communities must have similar
node counts or total node loads. This complete example requires a configured
solver. It asks for two nearly equal communities, keeps `"a"` and `"b"`
together, and prevents `"a"` and `"f"` from sharing a community.

```python
from collections import defaultdict

import networkx as nx

from asunder.load_balancing import LoadBalancer

graph = nx.Graph(
    [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("c", "d"),
        ("d", "e"),
        ("d", "f"),
        ("e", "f"),
    ]
)

result = LoadBalancer(
    graph,
    K=2,
    R=1,
    must_link=[("a", "b")],
    cannot_link=[("a", "f")],
    final_master_solve=True,
    disable_tqdm=True,
)

groups = defaultdict(list)
for node, community in result.metadata["community_map_labels"].items():
    groups[community].append(node)

print(dict(groups))
print("community loads:", result.metadata["community_balance_weights"])
print("modularity:", result.metadata["modularity"])
```

The numeric community labels are arbitrary; what matters is which nodes share
a label. `result.final_partition` is an `N x N` binary co-membership matrix in
the graph's node iteration order. It may be a dense Boolean array or a SciPy
CSR matrix, while retaining that same logical shape. Entry `[i, j]` is one when
nodes `i` and `j` belong to the same community. The label-aware metadata maps
that matrix back to the original NetworkX node labels.

Common controls include:

- `K` and `R` for the number of communities and the width of their permitted
  load range;
- `R_bounds=(lower, upper)` for explicit inclusive community-load bounds;
- `node_weight_attr="load"` to balance a positive integer node attribute
  instead of node count;
- `contract_graph=True` to contract must-linked nodes while preserving their
  summed balance weights;
- `resolution` to change modularity resolution; and
- `refine`, `use_refined_column`, `refine_post_loop`, `check_flat_pricing`, and
  `stopping_window` to control refinement and termination on large inputs.

Signed Leiden is the default pricing backend. QMETIS is an optional native
pricing heuristic selected with `algorithm="qmetis"` on a supported platform.
Its platform and approximation details are kept in the [QMETIS
reference](https://asunder.readthedocs.io/en/latest/reference/qmetis.html).

`LoadBalancer` raises `ValueError` for malformed inputs or impossible bound
definitions. It raises `RuntimeError` if the search does not produce an
integral feasible partition. Check bounds and pairwise constraints first, then
consider a larger search budget or `projection_repair=True`.

See the complete [load-balancing
guide](https://asunder.readthedocs.io/en/latest/getting_started/quickstart.html)
for weighted examples, result fields, and runtime controls.

## NLBNP structural workflows

`CorePeripheryPartition` is an NLBNP-specific shortcut for a constraint graph;
it is not presented as a general-purpose core-periphery partitioner. In its
intended use, the supplied grouping constraints collect the nonlinear nodes
into one detection block on the core side. The complementary periphery contains
only linear nodes. All of those periphery nodes are merged into final community
`0`, even when they are disconnected, which supplies the NLBNP requirement that
there be exactly one linear-only community.

The workflow then temporarily excludes that linear-only community from the
**original input graph** and computes connected components of the remaining
nonlinear/core-side induced subgraph, adding any supplied nonedge `must_link`
pairs as virtual edges. It does not perform this final component split on the
contracted detection graph. Those components are the independent communities;
all temporarily excluded nodes remain present in the returned labels and
metadata.

When `must_group` identifies designated nonlinear nodes, the workflow verifies
that they were detected on the core side. It raises `RuntimeError` rather than
returning a partition with the nonlinear block in the linear-only community.

Use this solver-free shortcut only when that NLBNP structure is appropriate and
the exposed components are already the desired independent communities. Use
`NonlinearBranchAndPrice` when the structural shortcut is insufficient and the
constraint graph needs the packaged column-generation workflow.

`NonlinearBranchAndPrice` requires at least one worthy edge from the input graph;
without an active edge rule, the problem is no longer NLBNP. It has three
cardinality modes. The default, `cardinality_method="reformulated"`, finds the
exact maximum linear-only set, reduces the result to pairwise constraints, and
enforces them alongside the edge-based constraint using column generation. An
explicit cannot-link inside that set is reported as infeasible. The
`"confidence"` and `"core_periphery"` modes first enforce the edge-based
constraint and then detect the linear-only group using confidence-score
clustering and core-periphery detection, respectively. All modes require the
nonlinear nodes, supplied directly or by a NetworkX node attribute.

The [NLBNP workflow
guide](https://asunder.readthedocs.io/en/latest/getting_started/nlbnp.html)
contains complete examples and explains how the three modes differ.

## Reusable decomposition and custom constraints

Use `run_csd_decomposition` when you need Asunder's orchestration but want to
supply or replace initial columns, the master problem, pricing, or refinement.
The [reusable decomposition
guide](https://asunder.readthedocs.io/en/latest/getting_started/base_decomposition.html)
defines those terms and provides a complete example.

ModularVFD refinement supports pairwise, component-local, community-wide, and
partition-wide hard constraints. Its [constraint-extension
guide](https://asunder.readthedocs.io/en/latest/reference/development/extending_modular_vfd.html)
shows how to implement them. A ModularVFD constraint governs ModularVFD
refinement only unless the same rule is also enforced in initial-column
generation, pricing, the master formulation, warm starts, and final validation.

For large sparse inputs, reusable decomposition and NLBNP preserve CSR through
preprocessing and compatible pricing. Hard columns use dense Boolean or CSR
Boolean storage according to measured density. Dense-only backends are guarded
by a configurable estimated working-set limit; see the [matrix-storage
reference](https://asunder.readthedocs.io/en/latest/reference/matrix_storage.html).

## Documentation and examples

- [Introduction](https://asunder.readthedocs.io/en/latest/getting_started/introduction.html)
- [Installation and solver setup](https://asunder.readthedocs.io/en/latest/getting_started/installation.html)
- [Load-balancing quickstart](https://asunder.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Reusable decomposition guide](https://asunder.readthedocs.io/en/latest/getting_started/base_decomposition.html)
- [NLBNP workflows](https://asunder.readthedocs.io/en/latest/getting_started/nlbnp.html)
- [API reference](https://asunder.readthedocs.io/en/latest/api/index.html)
- [Custom subproblem example](examples/custom_subproblem.py)
- [Nonlinear branch-and-price example](examples/nonlinear_bp.py)

Asunder does not automatically convert an optimization model into a graph. The
user supplies a NetworkX graph, an adjacency matrix, or application code that
constructs one. Before tuning algorithms, confirm that node identity, edge
meaning, edge weights, and hard constraints accurately represent the problem.
