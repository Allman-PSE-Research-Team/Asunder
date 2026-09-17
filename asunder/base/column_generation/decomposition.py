"""Column generation decomposition orchestration."""
from __future__ import annotations

import copy
import inspect
from collections import deque

import numpy as np
from scipy import sparse
from tqdm.auto import tqdm

from asunder.base.column_generation.master import compute_f_star
from asunder.base.utils.graph import (
    contract_adj_matrix_new,
    contract_node_pairs,
    contract_partition_matrix,
    expand_z_matrix,
    normalize_node_pairs,
    partition_satisfies_pairwise_constraints,
    sufficiently_different,
    validate_partition_matrix,
)
from asunder.base.utils.matrix import (
    DEFAULT_MAX_DENSE_WORKING_BYTES,
    DEFAULT_SPARSE_COLUMN_DENSITY_THRESHOLD,
    checked_to_dense,
    ensure_dense_working_set,
    is_symmetric,
    matrix_storage,
    normalize_adjacency,
    structural_edge_pairs,
    validate_storage_options,
)


def _weighted_column_sum(
    weights,
    columns,
    *,
    sparse_column_density_threshold,
    max_dense_working_bytes,
):
    """Build a fractional co-association matrix without accidental densification."""
    active = [
        (float(weight), column)
        for weight, column in zip(weights, columns)
        if not np.isclose(float(weight), 0.0)
    ]
    shape = columns[0].shape
    if not active:
        return sparse.csr_matrix(shape, dtype=float)

    all_sparse = all(sparse.issparse(column) for _, column in active)
    if all_sparse:
        result = sparse.csr_matrix(shape, dtype=float)
        for weight, column in active:
            values = sparse.csr_matrix(column, dtype=float)
            result = result + values.multiply(weight)
        result.sum_duplicates()
        result.eliminate_zeros()
        result.sort_indices()
        return result

    ensure_dense_working_set(
        shape,
        dtype=float,
        working_arrays=1.0,
        extra_bytes=2 * shape[1] * np.dtype(float).itemsize,
        max_dense_working_bytes=max_dense_working_bytes,
        operation="dense post-loop fractional co-association assembly",
    )
    result = np.zeros(shape, dtype=float)
    for weight, column in active:
        if sparse.issparse(column):
            values = sparse.csr_matrix(column)
            for row in range(shape[0]):
                start, end = values.indptr[row:row + 2]
                np.add.at(result[row], values.indices[start:end], weight * values.data[start:end])
        else:
            for row in range(shape[0]):
                result[row] += weight * column[row]
    return result


def CSD_decomposition(
    A, a, m,
    mp_function,
    sp_function,
    columns=None, f_stars=None,
    must_link=None, cannot_link=None,
    additional_constraints=None,
    contract_graph=False,
    stopping_window=5,
    check_flat_pricing=True,
    algo="signed_leiden",
    package="leidenalg",
    seed=42,
    # initial feasible column generator
    ifc_params: dict | None = None,
    # refinement
    refine_params: dict | None = None,
    use_refined_column=False,
    refine_post_loop=True,
    final_master_solve=True,
    max_iterations=1000, disable_tqdm=False,
    tolerance=1e-10, verbose=False,
    subproblem_params: dict | None = None,
    resolution=1.0,
    column_storage="auto",
    sparse_column_density_threshold=DEFAULT_SPARSE_COLUMN_DENSITY_THRESHOLD,
    max_dense_working_bytes=DEFAULT_MAX_DENSE_WORKING_BYTES,
    *,
    node_weights=None,
):
    """
    Function that does column generation (CG) and refinement given a master and subproblem function.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix, shape (N, N)
        Adjacency or weight matrix. Sparse input is normalized to CSR.
    a : np.ndarray of int | float, shape (N,)
        Degree-like vector; defaults to row sums of the symmetrized adjacency.
    m : float
        Twice the total weight in the graph.
    mp_function : callable
        Master problem function (Handles ILP and LP versions).
    sp_function : callable
        Pricing subproblem which can be implemented as a ILP or a heuristic (custom / third-party) subproblem.
    columns : list[numpy.ndarray or scipy.sparse.spmatrix] or None
        Existing binary co-association columns. This parameter is typically
        active during branch-and-price. With ``contract_graph=True``,
        original-dimension columns must respect every contracted component and
        are converted automatically. Sparse columns are normalized to CSR.
    f_stars : list[float] or None
        Objective values of the existing columns. This parameter is typically
        active during Branch and Price. Scores are recomputed on the contracted
        graph when contraction is active.
    must_link : list[tuple[int, int]]
        List of node pairs that must be together.
    cannot_link : list[tuple[int, int]]
        List of node pairs that must not be together.
    node_weights : array-like of float, shape (N,), optional
        Shared finite real node weights, defaulting to one per input node.
        Compatible hooks receive these as ``node_weights`` or
        ``balance_weights``. Contraction sums weights within each component;
        adjacency, degree vector ``a``, and modularity are not reweighted.
        Individual hooks may impose additional restrictions, such as positive
        integer loads. ``balance_weights`` also supplies this
        vector when ``node_weights`` is omitted. Repeated weight arguments
        must agree; application-specific vectors need distinct argument names.
    additional_constraints : dict[str, Any]
        Additional settings passed to the master, such as worthy edges or
        balance bounds. Non-pairwise hook settings must also be supplied in
        the corresponding generator, refinement, or subproblem arguments.
        Shared node weights are propagated separately.
    contract_graph : bool
        Whether must-links are handled by graph contraction. Cannot-links and
        initial-column constraints are mapped to component indices; a
        cannot-link internal to one component is rejected as infeasible.
    stopping_window : int
        Maximum number of allowed stagnant CG iterations. After this, CG is terminated.
    check_flat_pricing : bool
        Boolean that determines whether to check for flat/stagnant pricing or not.
    algo : str
        Name of heuristic subproblem used to replace the ILP subproblem. Third-party algorithms combine adjacency and dual information into a unified input while custom algorithms treat adjacency and duals as separate inputs. Supported third-party algorithms are listed under the ``package`` parameter.
        Available custom algorithm options include:

        ``"spectral"``:
            Modified iterative bisection algorithm based on Mark Newman's eigenvector-based method.
        ``"full_louvain"``:
            Modified but Louvain-like algorithm.
        ``"RCCS"``:
            This means Reduced Cost Community Search and is a greedy and local search heuristic for finding communities that maximize the reduced cost.
    package : str or None
        Package from which non-custom heuristic subproblem is selected. Package and algorithm options include:

        ``"networkx"``:
            ``"louvain"``, ``"greedy"``, ``"girvan_newman"``
        ``"sknetwork"``:
            ``"louvain"``, ``"leiden"``, ``"lpa"``
        ``"igraph"``:
            ``"leiden"``, ``"greedy"``, ``"infomap"``, ``"lpa"``, ``"multilevel"``, ``"voronoi"``, ``"walktrap"``, ``"cpm_leiden"``
        ``"leidenalg"``:
            ``"leiden"``,  ``"signed_leiden"``, ``"cpm_leiden"``, ``"surprise_leiden"``, ``"signed_surprise_leiden"``
        ``None``:
            ``"signed_louvain"``, ``"spinglass"``

        Algorithms that start with ``"cpm"``, ``"signed"``, and ``"spinglass"`` are signed.
    resolution : float, default=1.0
        Modularity resolution parameter. It is applied consistently when
        pricing and scoring columns.
    column_storage : {"auto", "dense", "csr"}, default="auto"
        Physical storage policy for binary co-association columns. Automatic mode
        preserves supplied dense/CSR representations, including mixed pools.
    sparse_column_density_threshold : float, default=0.20
        Maximum density for newly constructed automatic CSR columns; CSR must
        also have a smaller estimated footprint than dense Boolean storage.
    max_dense_working_bytes : int or None, default=536870912
        Maximum operation-specific estimate for package-created dense work
        arrays. Sparse-compatible operations retain CSR; dense-only boundaries
        raise ``MemoryError`` when the estimate is exceeded. ``None`` disables
        the guard, and existing dense caller input is not rejected merely
        because of its size.
    seed : int or None
        Random seed value.
    ifc_params : dict[str, callable or dict or int]
        Initial-column generator, column count, and arguments. Only
        ``must_link`` and ``cannot_link`` are reserved and injected into
        declared hook parameters. Standard weight arguments must match
        ``node_weights``; other settings remain caller-controlled.
    refine_params : dict[str, callable or dict]
        Refinement function and arguments. Pairwise constraints belong to the
        main workflow, not ``kwargs``. Balance settings, custom constraints,
        provenance, and search settings are allowed in ``kwargs``.
    subproblem_params : dict[str, Any] or None
        Pricing arguments, including non-pairwise constraint settings.
        Pairwise constraints enter through master duals, not these arguments.
    use_refined_column : bool
        Whether to run refinement and add its columns inside the main column
        generation loop.
    refine_post_loop : bool
        Boolean that determines whether a post-loop refinement column is generated after column generation terminates.
    final_master_solve : bool
        Boolean that determines whether a final master solve is executed or not.
    max_iterations : int
        Maximum number of column generation iterations.
    disable_tqdm : bool
        Whether to disable the progress bar or not.
    tolerance : float
        Tolerance value for terminating column generation.
    verbose : int or bool
        Controls the level of detail in the printed output.
        ``-1``: No output
        ``False`` | ``0``: Minimal output
        ``True`` | ``1``: Detailed output

    Returns
    -------
    results : list[dict]
        Column-generation iteration records. Each dictionary may include
        ``lambda_sol``, dual terms, ``master_obj_val``, ``z_sol``,
        ``sub_obj_val``, ``columns``, ``f_stars``, and ``heuristic_col``.
        Partition matrices may be dense Boolean arrays or Boolean CSR matrices.

        When graph contraction is active, public ``z_sol`` values are expanded
        to original-node dimensions. ``columns`` and ``heuristic_col`` remain
        in contracted component dimensions, and ``f_stars`` score those
        contracted columns. Each record also contains ``node2comp``, mapping
        original node indices to contracted component indices.

        If contraction leaves one component, the unique candidate is returned
        without calling the master, pricing, or refinement hooks.
    """
    validate_storage_options(
        column_storage,
        sparse_column_density_threshold,
        max_dense_working_bytes,
    )
    input_adjacency_storage = matrix_storage(A)
    if not sparse.issparse(A):
        A = checked_to_dense(
            A, max_dense_working_bytes=max_dense_working_bytes,
            operation="dense adjacency dtype normalization",
        )
    A = normalize_adjacency(A, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1)
    dense_boundary_events = []

    def normalize_column(candidate, *, name):
        return validate_partition_matrix(
            candidate,
            A.shape[0],
            name=name,
            storage=column_storage,
            sparse_column_density_threshold=sparse_column_density_threshold,
            max_dense_working_bytes=max_dense_working_bytes,
        )
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if a.shape != (A.shape[0],):
        raise ValueError(f"a must have shape {(A.shape[0],)}.")
    adjacency_finite = (
        np.all(np.isfinite(A.data)) if sparse.issparse(A)
        else all(np.all(np.isfinite(row)) for row in A)
    )
    if not adjacency_finite or not np.all(np.isfinite(a)):
        raise ValueError("A and a must contain only finite values.")
    if not is_symmetric(A, atol=1e-10):
        raise ValueError("A must be symmetric for undirected decomposition.")
    m = float(m)
    if not np.isfinite(m) or m <= 0:
        raise ValueError("m must be a finite positive value.")
    if max_iterations is not None and max_iterations < 1:
        raise ValueError("max_iterations must be at least 1 or None")
    ifc_params = copy.deepcopy(ifc_params or {})
    # Keep application-defined provenance identifiers intact.  In particular,
    # identity-hashable objects in ``component_members`` must remain the same
    # objects seen by constraint predicate closures.
    refine_params = dict(refine_params or {})
    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution < 0:
        raise ValueError("resolution must be a finite nonnegative value.")
    must_link = normalize_node_pairs(
        must_link,
        A.shape[0],
        relation_name="must-link",
    )
    cannot_link = normalize_node_pairs(
        cannot_link,
        A.shape[0],
        relation_name="cannot-link",
        reject_self=True,
    )
    contradictory_pairs = sorted(set(must_link) & set(cannot_link))
    if contradictory_pairs:
        raise ValueError(
            "The same node pair cannot be both must-link and cannot-link: "
            f"{contradictory_pairs}."
        )
    if stopping_window < 1:
        raise ValueError("stopping_window must be at least 1")

    # normalize refinement configurations
    refine_func = refine_params.get("refine_func")
    refine_kwargs = dict(refine_params.get("kwargs") or {})
    if callable(refine_func):
        refine_signature = inspect.signature(refine_func)
        if "gamma" in refine_signature.parameters:
            refine_kwargs.setdefault("gamma", resolution)
        if "max_dense_working_bytes" in refine_signature.parameters:
            refine_kwargs.setdefault(
                "max_dense_working_bytes", max_dense_working_bytes
            )

    if use_refined_column and not callable(refine_func):
        raise ValueError(
            "use_refined_column=True requires refine_params['refine_func']."
        )

    # Make generator arguments optional.
    ifc_params["args"] = dict(ifc_params.get("args") or {})

    # drop seed values from parameters if included
    ifc_params["args"].pop("seed", None)
    refine_kwargs.pop("seed", None)
    subproblem_params = (
        {} if subproblem_params is None else dict(subproblem_params)
    )
    owned_constraint_keys = {"must_link", "cannot_link"}
    for name, options in (
        ("ifc_params['args']", ifc_params["args"]),
        ("refine_params['kwargs']", refine_kwargs),
        ("subproblem_params", subproblem_params),
    ):
        forbidden = owned_constraint_keys.intersection(options)
        if forbidden:
            raise ValueError(
                f"{name} cannot contain workflow-owned pairwise constraints: "
                f"{', '.join(sorted(forbidden))}. Set must_link/cannot_link "
                "on the main workflow instead."
            )

    # validate warm start when necessary
    columns = [] if columns is None else list(columns)
    f_stars = None if f_stars is None else list(f_stars)

    if columns and f_stars is None:
        raise ValueError(
            "Warm-start f_stars are required when columns are supplied."
        )

    if f_stars is not None and len(columns) != len(f_stars):
        raise ValueError(
            "Warm-start columns and f_stars must have the same length."
        )

    additional_constraints = (
        {} if additional_constraints is None else dict(additional_constraints)
    )
    # normalize node weights
    if node_weights is None:
        node_weights = additional_constraints.get("node_weights")
    if node_weights is None:
        node_weights = additional_constraints.get("balance_weights")
    node_weights = (
        np.ones(A.shape[0], dtype=float) if node_weights is None
        else np.asarray(node_weights, dtype=float)
    )
    if node_weights.shape != (A.shape[0],) or not np.all(np.isfinite(node_weights)):
        raise ValueError("node_weights must contain one finite real value per input node.")

    weight_options = (
        ("additional_constraints", mp_function, additional_constraints),
        ("ifc_params['args']", ifc_params.get("generator"), ifc_params["args"]),
        ("refine_params['kwargs']", refine_func, refine_kwargs),
        ("subproblem_params", sp_function, subproblem_params),
    )
    for name, hook, options in weight_options:
        parameters = inspect.signature(hook).parameters if callable(hook) else {}
        for key in ("node_weights", "balance_weights"):
            if key not in options and key not in parameters:
                continue
            supplied = options.get(key)
            if supplied is not None and not np.array_equal(
                np.asarray(supplied, dtype=float), node_weights
            ):
                raise ValueError(
                    f"{name}['{key}'] must match the main node_weights. "
                    "Use a distinct argument name for application-specific weights."
                )
            options[key] = node_weights
    if additional_constraints.get("LB"):
        additional_constraints["balance_weights"] = node_weights
    refinement_must_link = list(must_link)
    worthy_edges = additional_constraints.get("worthy_edges")
    if worthy_edges is not None:
        worthy = {tuple(sorted(edge)) for edge in worthy_edges}
        refinement_must_link = sorted(set(refinement_must_link).union(
            pair for pair in structural_edge_pairs(A) if pair not in worthy
        ))
    # Only pairwise constraints are owned by CSD. Pricing receives these via
    # master duals; generators and refiners receive their declared arguments.
    for hook, options in (
        (ifc_params.get("generator"), ifc_params["args"]),
        (refine_func, refine_kwargs),
    ):
        if callable(hook):
            parameters = inspect.signature(hook).parameters
            if "must_link" in parameters:
                options["must_link"] = list(refinement_must_link)
            if "cannot_link" in parameters:
                options["cannot_link"] = list(cannot_link)

    # contract graph if necessary
    edge_constraints_active = (
        "worthy_edges" in additional_constraints
        and additional_constraints["worthy_edges"] is not None
    )
    if contract_graph and (must_link or edge_constraints_active):
        A, node2comp = contract_adj_matrix_new(
            A, additional_constraints.get("worthy_edges"), must_link,
            max_dense_working_bytes=max_dense_working_bytes,
        )
        a = np.asarray(A.sum(axis=0)).reshape(-1)
        m = float(np.sum(a))
        if edge_constraints_active:
            additional_constraints["worthy_edges"] = None
        node_weights = np.bincount(
            node2comp,
            weights=node_weights,
            minlength=A.shape[0],
        )
        for _, _, options in weight_options:
            for key in ("node_weights", "balance_weights"):
                if key in options:
                    options[key] = node_weights
        cannot_link = contract_node_pairs(
            cannot_link,
            node2comp,
            relation_name="cannot-link",
            reject_internal=True,
        )

        generator_args = ifc_params.get("args", {})
        if "N" in ifc_params.get("args", {}):
            generator_args["N"] = np.shape(A)[0]
        if "G" in generator_args:
            import networkx as nx

            generator_args["G"] = (
                nx.from_scipy_sparse_array(A)
                if sparse.issparse(A)
                else nx.from_numpy_array(A)
            )
            generator_args["nodes"] = list(range(A.shape[0]))
        if "cannot_link" in generator_args:
            generator_args["cannot_link"] = contract_node_pairs(
                generator_args["cannot_link"],
                node2comp,
                relation_name="initial-column cannot-link",
                reject_internal=True,
            )
        if "must_link" in generator_args:
            generator_args["must_link"] = []

        if "cannot_link" in refine_kwargs:
            refine_kwargs["cannot_link"] = cannot_link
        if "must_link" in refine_kwargs:
            refine_kwargs["must_link"] = []
        for options in (generator_args, refine_kwargs, subproblem_params):
            hook_edges = options.get("worthy_edges")
            if (
                edge_constraints_active and hook_edges is not None
                and {tuple(sorted(edge)) for edge in hook_edges} == worthy
            ):
                # This identical edge rule is already enforced by contraction.
                options["worthy_edges"] = None
        if callable(refine_func) and "component_members" in refine_signature.parameters:
            supplied_members = refine_kwargs.get("component_members")
            if supplied_members is None:
                source_members = [(int(i),) for i in range(node2comp.shape[0])]
            else:
                if len(supplied_members) != node2comp.shape[0]:
                    raise ValueError(
                        "component_members must contain one member collection per "
                        "node before external contraction."
                    )
                source_members = [tuple(members) for members in supplied_members]

            contracted_members = [[] for _ in range(A.shape[0])]
            for source_index, component_index in enumerate(node2comp):
                contracted_members[int(component_index)].extend(source_members[source_index])
            refine_kwargs["component_members"] = tuple(
                tuple(members) for members in contracted_members
            )

        if columns is not None and len(columns) > 0:
            columns = [
                normalize_column(
                    contract_partition_matrix(column, node2comp, max_dense_working_bytes=max_dense_working_bytes),
                    name=f"contracted warm-start column {index}",
                )
                for index, column in enumerate(columns)
            ]
            # Recompute scores so the column pool and contracted objective
            # cannot retain mismatched original-graph bookkeeping.
            f_stars = [
                compute_f_star(A, a, m, column, gamma=resolution)
                for column in columns
            ]

        if A.shape == (1, 1):
            if additional_constraints.get("LB"):
                requested_k = int(additional_constraints.get("K", 1))
                if requested_k != 1:
                    return None
                total_weight = float(node_weights[0])
                bounds = additional_constraints.get("R_bounds")
                if bounds is None:
                    width = int(additional_constraints.get("R", 0))
                    lower = max(1, int(np.floor(total_weight - width / 2 + 0.5)))
                    upper = lower + width
                else:
                    lower, upper = bounds
                if not (float(lower) <= total_weight <= float(upper)):
                    return None
            coarse_partition = validate_partition_matrix(
                np.ones((1, 1), dtype=bool),
                storage=column_storage,
                sparse_column_density_threshold=sparse_column_density_threshold,
                max_dense_working_bytes=max_dense_working_bytes,
            )
            score = (
                compute_f_star(A, a, m, coarse_partition, gamma=resolution)
                if m > 0
                else 0.0
            )

            print(
                "The graph contracted to one component. Only inexpensive "
                "built-in checks were performed; the configured master, "
                "pricing, and column-refinement hooks were skipped. "
                "Custom feasibility was not checked."
            )
            return [{
                "lambda_sol": [1.0],
                "master_obj_val": score,
                "z_sol": expand_z_matrix(coarse_partition, node2comp, max_dense_working_bytes=max_dense_working_bytes),
                "heuristic_col": None,
                "sub_obj_val": None,
                "columns": [coarse_partition],
                "f_stars": [score],
                "node2comp": node2comp.copy(),
                "partition_source": "contracted_trivial",
                "storage_metadata": {
                    "input_adjacency_storage": input_adjacency_storage,
                    "working_adjacency_storage": matrix_storage(A),
                    "column_storage": column_storage,
                    "sparse_column_density_threshold": float(
                        sparse_column_density_threshold
                    ),
                    "max_dense_working_bytes": max_dense_working_bytes,
                    "column_storage_counts": {
                        matrix_storage(coarse_partition): 1,
                    },
                    "dense_boundary_events": (),
                },
            }]
    else:
        node2comp = None

    if not columns:
        missing = {"generator", "num"} - ifc_params.keys()
        if missing:
            raise ValueError(
                f"ifc_params is missing required keys: {sorted(missing)}"
            )

        if not callable(ifc_params["generator"]):
            raise ValueError("ifc_params['generator'] must be callable.")

        if ifc_params["num"] < 1:
            raise ValueError("ifc_params['num'] must be at least 1.")

        generator_signature = inspect.signature(ifc_params["generator"])
        generator_options = {
            "column_storage": column_storage,
            "sparse_column_density_threshold": sparse_column_density_threshold,
            "max_dense_working_bytes": max_dense_working_bytes,
        }
        for option, value in generator_options.items():
            if option in generator_signature.parameters:
                ifc_params["args"].setdefault(option, value)

    # deque to track flat pricing for termination
    SUB_OBJS = deque(maxlen=stopping_window)

    if (columns is not None and f_stars is not None) and len(columns) > 0:
        # initialize from parameters
        Z_star = [
            normalize_column(column, name=f"warm-start column {index}")
            for index, column in enumerate(columns)
        ]
        if resolution != 1.0:
            # Caller-supplied scores do not record the resolution at which
            # they were computed, so non-default runs rescore their columns.
            f_stars = [
                compute_f_star(A, a, m, column, gamma=resolution)
                for column in Z_star
            ]
    else:
        # generate initial feasible columns
        ifc_generator = ifc_params["generator"]
        feasible_columns = ifc_generator(**ifc_params["args"], seed=seed)

        # without feasible columns, terminate
        if feasible_columns is None or len(feasible_columns) == 0:
            if verbose != -1:
                print("A feasible initial partition cannot be generated.")
            return None

        # initialize columns and their scores
        if ifc_params["num"] == 1:
            initial_z = normalize_column(
                feasible_columns[0], name="initial feasible column 0"
            )
            f_star_initial = compute_f_star(
                A, a, m, initial_z, gamma=resolution
            )
            Z_star = [initial_z]
            f_stars = [f_star_initial]
        else:
            Z_star = [
                normalize_column(
                    column, name=f"initial feasible column {index}"
                )
                for index, column in enumerate(feasible_columns[:ifc_params["num"]])
            ]
            f_stars = [
                compute_f_star(A, a, m, col, gamma=resolution)
                for col in Z_star
            ]

    results = []

    # main column generation loop
    # for iteration in tqdm(range(max_iterations)):
    with tqdm(total=max_iterations, disable=disable_tqdm) as pbar:
        iteration = 1

        while True:
            if verbose != -1:
                print("\nIteration:", iteration)

            # run relaxed master problem (RMP)
            (lambda_sol,
            duals,
            master_obj_val
            ) = mp_function(
                A, a, m,
                Z_star, f_stars,
                cannot_link=cannot_link,
                must_link=[] if contract_graph else must_link,
                **additional_constraints,
                verbose=verbose,
                extract_dual=True,
            )

            # call it a day if RMP is infeasible
            if master_obj_val is None:
                return None

            if sparse.issparse(A) and any(
                isinstance(dual, np.ndarray) and dual.ndim in {1, 2}
                for dual in duals.values()
                if dual is not None
            ):
                event = "pricing:dense_duals"
                if event not in dense_boundary_events:
                    dense_boundary_events.append(event)

            if verbose != -1:
                print(duals)

                print("Master Obj:", master_obj_val)
                print("lambda: ", lambda_sol)

            pricing_kwargs = {
                **subproblem_params,
                "algo": algo,
                "package": package,
                "gamma": resolution,
                "verbose": verbose,
                "seed": seed,
                "column_storage": column_storage,
                "sparse_column_density_threshold": sparse_column_density_threshold,
                "max_dense_working_bytes": max_dense_working_bytes,
            }
            signature = inspect.signature(sp_function)
            accepts_extra = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if resolution != 1.0 and not accepts_extra and "gamma" not in signature.parameters:
                raise ValueError(
                    f"{getattr(sp_function, '__name__', 'The selected subproblem')} "
                    "does not support a non-default resolution."
                )
            if not accepts_extra:
                pricing_kwargs = {
                    name: value
                    for name, value in pricing_kwargs.items()
                    if name in signature.parameters
                }
            sub_obj_val, z_sol = sp_function(
                A,
                a,
                m,
                duals,
                **pricing_kwargs,
            )
            pricing_name = getattr(sp_function, "__name__", "")
            if sparse.issparse(A) and (
                pricing_name == "custom_heuristic_subproblem"
                or (pricing_name == "heuristic_subproblem" and algo == "signed_louvain")
            ):
                event = f"pricing:{algo}"
                if event not in dense_boundary_events:
                    dense_boundary_events.append(event)
            z_sol = normalize_column(z_sol, name="pricing column")

            results.append({
                "lambda_sol": lambda_sol,
                **duals,
                "master_obj_val": master_obj_val,
                "z_sol": expand_z_matrix(z_sol, node2comp, max_dense_working_bytes=max_dense_working_bytes) if contract_graph else z_sol,
                "sub_obj_val": sub_obj_val, "columns": Z_star.copy(),
                "f_stars": f_stars.copy(),
                "partition_source": "pricing_candidate",
            })

            # check if the pricing problem generated a column with positive reduced cost.
            if sub_obj_val > tolerance: # and iteration == 0:
                if verbose != -1:
                    print("New column generated with f* =", sub_obj_val)
                    print(z_sol)
            else:
                if verbose != -1:
                    print(f"No improving column found (reduced cost: {sub_obj_val:.2g}); stopping column generation.")
                break

            # add priced out column and its score to the column set
            Z_star.append(z_sol)
            f_stars.append(
                compute_f_star(A, a, m, z_sol, gamma=resolution)
            )
            SUB_OBJS.append(sub_obj_val)

            # refine z_sol and potentially add to column list
            if callable(refine_func) and use_refined_column:
                # refine column only when explicitly enabled in-loop.
                in_loop_kwargs = dict(refine_kwargs)
                if "shake_rounds" in inspect.signature(refine_func).parameters:
                    in_loop_kwargs["shake_rounds"] = 0
                in_loop_kwargs.setdefault("seed", seed)
                heuristic_col = refine_func(
                    A=A,
                    partition=z_sol,
                    **in_loop_kwargs,
                )
                if sparse.issparse(A) or sparse.issparse(z_sol):
                    refiner_name = getattr(refine_func, "__name__", "")
                    if refiner_name in {
                        "refine_partition_modular_vfd",
                        "refine_partition_with_cp",
                    }:
                        event = f"refinement:{refiner_name}"
                        if event not in dense_boundary_events:
                            dense_boundary_events.append(event)
                if heuristic_col is not None:
                    heuristic_col = normalize_column(
                        heuristic_col, name="in-loop refinement column"
                    )
                    if not partition_satisfies_pairwise_constraints(
                        heuristic_col,
                        must_link=[] if contract_graph else refinement_must_link,
                        cannot_link=cannot_link,
                    ):
                        heuristic_col = None
                if heuristic_col is not None:
                    results[-1]["heuristic_col"] = heuristic_col
                    # add heuristic column to list if it is needed in column generation and is sufficiently different
                    if use_refined_column and sufficiently_different(heuristic_col, Z_star, dist_min=0.01):
                        Z_star.append(heuristic_col)
                        f_stars.append(
                            compute_f_star(
                                A,
                                a,
                                m,
                                heuristic_col,
                                gamma=resolution,
                            )
                        )

            # if pricing is flat for `stopping_window` iterations, stop.
            if (
                check_flat_pricing
                and len(SUB_OBJS) == stopping_window
                and (max(SUB_OBJS) - min(SUB_OBJS)) <= tolerance
            ):
                if verbose != -1:
                    print(f"Pricing seems flat after {stopping_window} iterations; Stopping column generation...")
                break

            iteration += 1
            pbar.update(1)
            if max_iterations is not None:
                if iteration > max_iterations:
                    break
    # generate heuristic column using wz after CG terminates and add it to results
    if refine_post_loop and callable(refine_func):
        if not disable_tqdm:
            print("Running post-loop refinement...")
        wz = _weighted_column_sum(
            lambda_sol,
            Z_star,
            sparse_column_density_threshold=sparse_column_density_threshold,
            max_dense_working_bytes=max_dense_working_bytes,
        )

        post_loop_kwargs = dict(refine_kwargs)
        post_loop_kwargs.setdefault("seed", seed)
        heuristic_col = refine_func(
            A=A,
            partition=wz,
            **post_loop_kwargs,
        )
        if sparse.issparse(A) or sparse.issparse(wz):
            refiner_name = getattr(refine_func, "__name__", "")
            if refiner_name in {
                "refine_partition_modular_vfd",
                "refine_partition_with_cp",
            }:
                event = f"refinement:{refiner_name}"
                if event not in dense_boundary_events:
                    dense_boundary_events.append(event)
        if heuristic_col is not None:
            heuristic_col = normalize_column(
                heuristic_col, name="post-loop refinement column"
            )
            if not partition_satisfies_pairwise_constraints(
                heuristic_col,
                must_link=[] if contract_graph else refinement_must_link,
                cannot_link=cannot_link,
            ):
                heuristic_col = None
        if heuristic_col is not None:
            Z_star.append(heuristic_col)
            f_stars.append(
                compute_f_star(
                    A,
                    a,
                    m,
                    heuristic_col,
                    gamma=resolution,
                )
            )
            empty_duals = {k:None for k,v in duals.items()}

            results.append({
                "lambda_sol": None,
                **empty_duals,
                "master_obj_val": None,
                "z_sol": expand_z_matrix(heuristic_col, node2comp, max_dense_working_bytes=max_dense_working_bytes) if contract_graph else heuristic_col,
                "heuristic_col": heuristic_col,
                "sub_obj_val": None,
                "columns": Z_star.copy(),
                "f_stars": f_stars.copy(),
                "partition_source": "post_loop_refinement",
            })

    if final_master_solve:
        if verbose != -1:
            print("Final Integer Master Solve...")
        (lambda_sol, master_obj_val) = mp_function(
            A, a, m,
            Z_star, f_stars,
            cannot_link=cannot_link,
            must_link=[] if contract_graph else must_link,
            **additional_constraints,
            verbose=verbose,
            extract_dual=False,
        )
        if lambda_sol is not None:
            z_sol = Z_star[np.argmax(lambda_sol)]
        else:
            if verbose != -1:
                print("Final master solve is infeasible.")
            z_sol = None

        empty_duals = {k:None for k, _ in duals.items()}

        results.append({
            "lambda_sol": lambda_sol,
            **empty_duals,
            "master_obj_val": master_obj_val,
            "z_sol": expand_z_matrix(z_sol, node2comp, max_dense_working_bytes=max_dense_working_bytes) if contract_graph else z_sol,
            "heuristic_col": None,
            "sub_obj_val": None,
            "columns": Z_star.copy(),
            "f_stars": f_stars.copy(),
            "partition_source": "integer_master",
        })
    if node2comp is not None:
        for record in results:
            record["node2comp"] = node2comp.copy()
    if results:
        counts = {"dense": 0, "csr": 0}
        for column in Z_star:
            counts[matrix_storage(column)] += 1
        results[-1]["storage_metadata"] = {
            "input_adjacency_storage": input_adjacency_storage,
            "working_adjacency_storage": matrix_storage(A),
            "column_storage": column_storage,
            "sparse_column_density_threshold": float(
                sparse_column_density_threshold
            ),
            "max_dense_working_bytes": max_dense_working_bytes,
            "column_storage_counts": counts,
            "dense_boundary_events": tuple(dense_boundary_events),
        }
    return results
