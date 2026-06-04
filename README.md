![](assets/asunder.gif)
---
Asunder is a Python package for constrained network structure detection (constrained graph clustering) on undirected graphs, with a workflow centered on column generation and customizable master/subproblem pipelines. Graph clustering itself is an important task in a lot of traditional optimization, data-mining and machine learning pipelines. In these application areas, constraints on the kind of clusters or structures that are detected naturally occur but generalized workflows / packages for handling them did not exist. Asunder changes that.

Asunder works by combining traditional solver based optimization with classical machine learning algorithms. In the process, expensive Integer Linear Program (ILP) subproblems are replaced with heuristic clustering algorithms while ensuring that dual information from an LP master problem are respected. This enables the solution of a wide range of constrained structure detection (constrained graph clustering) problems, insofar as a master problem, and any other relevant custom element, can be properly formulated. See the [problem fit section](#problem-fit) and the [Asunder documentation](https://asunder.readthedocs.io/) for more details.

For users who want a pre-configured load balancing workflow, Asunder includes `asunder.load_balancing.LoadBalancer`: a high-level load-balanced graph partitioning workflow with built-in initial column generation, master problem handling, and refinement.

Development of Asunder is led by [Andrew Allman's Process Systems Research Team](https://allmanaa.engin.umich.edu/) at the University of Michigan.

## Install

Base install:

```bash
python3 -m pip install put-asunder
```

Optional extras:

```bash
python3 -m pip install "put-asunder[graph,viz]"
```

Legacy heuristics (best-effort on Python 3.13):

```bash
python3 -m pip install "put-asunder[legacy]"
```

## Python Support

- Guaranteed: Python 3.10, 3.11, 3.12, 3.13 for core package.
- Guaranteed: mainstream extras (`graph`, `viz`) on Python 3.10 to 3.13.
- Best-effort: `legacy` extra on Python 3.13.

## Package Layout

- `asunder`: top-level facade for orchestration, config, solvers, and common convenience entry points.
- `asunder.base`: reusable algorithms, branch-and-price utilities, column-generation modules, metrics, utilities, and visualization helpers.
- `asunder.load_balancing`: the built-in load balancing application layer, including load-balanced initial feasible column generation, master problem handling, and refinement.
- `asunder.nlbnp`: the nonlinear branch-and-price application layer, including a generic high-level workflow, case studies, evaluation flow, and NLBNP-specific refinement.

## Quickstart

```python
import numpy as np
from asunder import CSDDecomposition, CSDDecompositionConfig

# graph adjacency
A = np.array([
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
], dtype=float)

# ifc_params contains function and parameters for generating initial feasible partition(s)
cfg = CSDDecompositionConfig(
    ifc_params={"generator": lambda N, **_: [np.ones((N, N))], "num": 1, "args": {"N": A.shape[0]}},
    extract_dual=False,
    final_master_solve=False,
)

result = CSDDecomposition(config=cfg).run(A)
print(result.metadata)
```

The example above uses the top-level facade. Canonical reusable imports live under `asunder.base`, the packaged load balancing workflow lives under `asunder.load_balancing`, and the packaged nonlinear branch-and-price workflow lives under `asunder.nlbnp`.

```python
from asunder.base.column_generation.subproblem import heuristic_subproblem
from asunder.base.algorithms.modular_VFD import modular_very_fortunate_descent
from asunder.nlbnp import CorePeripheryPartition, NonlinearBranchAndPrice
from asunder.nlbnp.algorithms.refinement import refine_partition_linear_group, refine_partition_with_cp
from asunder.nlbnp.case_studies import run_evaluation
```

### Load balancing
The packaged load balancing workflow is the fastest path when you need graph partitions whose community sizes are fixed or bounded. It accepts a `networkx.Graph`, optional `must_link` and `cannot_link` pairs written in the graph's node labels, and either a target number of communities `K` with tolerance `R` or explicit `R_bounds`.

```python
import numpy as np
import networkx as nx
from asunder.load_balancing import LoadBalancer

A = np.array([
    [0, 1, 1, 1, 1, 1, 0, 0],
    [1, 0, 1, 1, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0]
], dtype=float)

G = nx.from_numpy_array(A)

z, metadata = LoadBalancer(G, K=2, R=0, disable_tqdm=True)
print(f'Detected the partition below with a modularity of {metadata["modularity"]} in {metadata["execution_time"]:.2f} secs')
print(z)
```

`z` is the detected partition matrix. `metadata` includes the modularity score, elapsed time, and label-aware community information so the result can be mapped back to the original graph nodes.

### Nonlinear branch-and-price
The generic nonlinear branch-and-price workflow lives under `asunder.nlbnp` and accepts user-provided graphs instead of case-study names.

```python
import networkx as nx
from asunder.nlbnp import NonlinearBranchAndPrice

G = nx.Graph()
G.add_edge("u", "v", edge_kind="integer")
G.add_edge("v", "w", edge_kind="continuous")

result = NonlinearBranchAndPrice(
    G,
    worthy_edge_attr="edge_kind",
    worthy_edge_value="integer",
    algorithm="louvain",
    package="networkx",
    disable_tqdm=True,
)

print(result.final_partition)
print(result.metadata["community_map_labels"])
```

For a faster component-level solution, use `CorePeripheryPartition`. It detects
a core, removes it, and treats each remaining connected component as a final
community:

```python
community_labels, metadata = CorePeripheryPartition(
    G,
    unworthy_edge_attr="edge_kind",
    unworthy_edge_value="continuous",
    cp_algorithm="SPEC",
)
```

Use `CorePeripheryPartition` when each connected component left after core
removal is already an appropriate final community. Use `NonlinearBranchAndPrice` 
when the community structure is beyond the direct core-periphery logic. The
column-generation workflow can also use `refine_partition_with_cp` through its
generic `refine_params` hook.

Use `run_evaluation` only when you want the packaged benchmark/case-study evaluation flow.

## Solver Setup

Asunder accepts user-provided solver objects. Solver support is configured through your local environment rather than through a dedicated package extra. For Gurobi, `GRB_LICENSE_FILE` is used by your environment. Example:

```python
from asunder import create_solver

solver = create_solver("gurobi_direct")
```

## Problem Fit

Asunder supports general constrained partitioning when requirements can be expressed as:

- load balancing constraints
- community size constraints
- must-link and cannot-link constraints
- edge-based constraints

Asunder works well out of the box when load balancing is needed; start with `asunder.load_balancing.LoadBalancer` for that case. Asunder also works well for optimization problems where coordination or operations are coupled across space (e.g. central coupling) and/or time and those interactions can be represented as a graph over constraints.

Sample fit signals:

- load balancing: graph partitions must have equal, near-equal, or explicitly bounded community sizes.
- load balancing: must-link or cannot-link pairs express operational grouping rules between graph nodes.
- reusable decomposition: you need to provide custom initial columns, master logic, subproblem logic, or refinement.
- nonlinear branch and price: coupling across time periods, units, or resources creates meaningful constraint interactions.
- nonlinear branch and price: there is value from multilevel partitioning or core-periphery structure detection.
- core-periphery partitioning: removing a central core leaves connected components that are meaningful final communities.

Some representative domains:

- load balancing for decomposition workloads, service territories, team assignment, scenario grouping, and graph-backed resource allocation.
- nonlinear branch and price for stochastic design and dispatch in energy systems.
- nonlinear branch and price for scheduling and resource allocation in healthcare systems.
- nonlinear branch and price for planning, routing, and location in supply chain and logistics.
- nonlinear branch and price for network configuration and resource management in telecommunications.

As a rule of thumb, reusable decomposition logic belongs in `asunder.base`, load balancing application logic belongs in `asunder.load_balancing`, and nonlinear branch-and-price application logic belongs in `asunder.nlbnp`.

For a fuller guide on where default workflows are sufficient versus where customization helps, see the [problem fit guide](https://asunder.readthedocs.io/en/latest/learn/guides/problem_fit.html).

## Customization Points

For custom problems, typical extension points are:

1. Initial feasible partition generator.
2. `solve_master_problem` replacement.
3. Optional heuristic or ILP subproblem replacement.
4. Optional partition refinement stage.

Reusable extension logic should generally be added under `asunder.base`. `asunder.load_balancing`, for instance, uses modules from `asunder.base`, but defines application specific modules separately. The built-in nonlinear branch-and-price refinement path lives under `asunder.nlbnp.algorithms`.

## Constraint Graph Compatibility

Required structure for `asunder.load_balancing.LoadBalancer`:
- undirected graph (`networkx.Graph`)
- node IDs that can be mapped back to the original system; `must_link` and `cannot_link` constraints should use those node labels
- optional application-specific node attributes; the high-level workflow does not require the NLBNP case-study schema
- size controls through `K` and `R`, or explicit lower/upper community-size bounds through `R_bounds`

Required structure for `run_evaluation`-style workflows:

For the built-in case-study evaluation workflows (`run_evaluation`, implemented in `asunder.nlbnp.case_studies.runner` and re-exported at top level), Asunder expects a constraint-graph pattern consistent with the provided case studies.

- undirected graph (`networkx.Graph`)
- node attribute `constraint` (string tag used for ground-truth and role grouping)
- edge attribute `var_type` with values `"integer"` or `"continuous"`

Commonly present (recommended) attributes:

- node attribute `type` (for example `"constraint"`)
- node attribute `details` (metadata dict)
- edge attributes `weight`, `variables`, `var_types`

How these are used:

- `constraint` identifies core/nonlinear tags in built-in case studies
- `var_type` determines candidate edge sets for core-periphery (CP) and community detection with refinement (CD_Refine) paths

If you are not using `run_evaluation`, use `asunder.nlbnp.CorePeripheryPartition` for component-level communities after core removal or `asunder.nlbnp.NonlinearBranchAndPrice` when the community structure is beyond the direct core-periphery logic. For lower-level customization, call the decomposition APIs directly with an adjacency matrix plus explicit constraints.

## Examples

- Load balancing quickstart: [docs/getting_started/quickstart.rst](docs/getting_started/quickstart.rst)
- Nonlinear B&P-style decomposition: `examples/nonlinear_bp.py`
- Custom subproblem wiring: `examples/custom_subproblem.py`

## Documentation

The full documentation is available at [asunder.readthedocs.io](https://asunder.readthedocs.io/). That includes getting started guides, problem-fit guidance, API reference pages for `asunder.base`, `asunder.load_balancing`, `asunder.nlbnp`, and development notes.

## References

Asunder integrates or wraps methods from:

- `networkx`
- `sklearn`
- `python-igraph` / `leidenalg`
- `scikit-network`
- `signed-louvain` style algorithms

`python-igraph` / `leidenalg` are best for massive networks (millions of edges) requiring highly optimized C++ speeds. `networkx` is best for quick prototyping, small-to-medium networks, and deep integration with native Python environments. See [documentation](https://asunder.readthedocs.io/) to discover what algorithms are available from each package.
