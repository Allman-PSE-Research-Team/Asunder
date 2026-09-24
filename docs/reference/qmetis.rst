QMETIS Backend Reference
========================

QMETIS is an optional native pricing heuristic for the load-balancing
workflow. Most users can select it with ``algorithm="qmetis"`` and leave its
lower-level settings unchanged.

High-level use
--------------

This example requires a supported platform wheel, a configured optimization
solver, and a complete input graph:

.. code-block:: python

   import networkx as nx

   from asunder.load_balancing import LoadBalancer

   graph = nx.cycle_graph(8)
   result = LoadBalancer(
       graph,
       K=2,
       R=1,
       algorithm="qmetis",
       final_master_solve=True,
       disable_tqdm=True,
   )

   print(result.metadata["community_map_labels"])
   print(result.metadata["qmetis_release"])

Platform packaging
------------------

Released Windows x86-64, Linux x86-64, and macOS universal2 wheels bundle the
pinned ``qmetis-v5.2.1-modularity.3`` native library with an ``idx64-real32``
ABI. Each wheel contains the appropriate ``qmetis.dll``, ``libqmetis.so``, or
``libqmetis.dylib``. Generic ``metis``-named libraries are deliberately
excluded.

The source distribution contains no native library. Other Asunder workflows
remain available from a source installation, but selecting QMETIS requires a
compatible library to be staged during a platform-wheel build.

Graph and matrix interfaces
---------------------------

:func:`~asunder.load_balancing.algorithms.qmetis.qmetis_load_balanced_partition`
accepts a NetworkX graph. ``node_weight_attr`` identifies positive integer
node loads, and ``edge_weight_attr`` identifies integer graph-edge weights.

:func:`~asunder.load_balancing.algorithms.qmetis.run_qmetis` accepts a matrix.
Matrix entries are already the edge weights, so it does not accept an edge
attribute name. Use its ``node_weights`` vector for balance loads; node
attribute names are not meaningful for matrix input.

The lower-level
:func:`~asunder.load_balancing.algorithms.qmetis.qmetis_part_graph` accepts a
stable weighted adjacency-list representation and forwards native METIS-style
options.

Quantization and exact rescoring
--------------------------------

QMETIS consumes integer edge weights, while Asunder pricing commonly produces
fractional dual-adjusted weights. The wrapper therefore:

1. clips the QMETIS search matrix to nonnegative values;
2. quantizes it within ``relative_resolution`` and ``safe_total`` limits;
3. asks QMETIS for a candidate partition; and
4. recomputes reduced cost using the original floating-point adjacency and
   dual values.

The native objective guides candidate generation but is not treated as the
exact Asunder reduced cost. ``resolution`` is Asunder's modularity resolution
parameter and is forwarded consistently to candidate generation and exact
rescoring.

Diagonal and contraction behavior
---------------------------------

QMETIS does not reliably support adjacency self-loops. The candidate generator
drops nonzero diagonal entries and emits
:class:`~asunder.load_balancing.algorithms.qmetis.QMETISApproximationWarning`.
This matters especially after graph contraction, where diagonal entries can
contain internal edge mass.

Dropping those values affects only the QMETIS search approximation. Exact
reduced-cost rescoring continues to use the original matrix, including its
diagonal. A candidate can therefore be useful even when the native QMETIS
objective and the exact reduced cost differ.

Load balance
------------

The load-balancing pricing adapter derives QMETIS's imbalance tolerance from
Asunder's K/R or explicit-bound semantics. Matrix-level ``node_weights`` must
contain one finite positive integer per row. Graph-level node attributes also
must be finite positive integers; a missing selected attribute defaults to
one.

Automatic quantization applies only to edge weights, not node loads. Convert
fractional loads and explicit load bounds to common integer units yourself;
see :doc:`load_balancing` for scaling and weighted K/R semantics. Changing only
the pricing backend does not remove the integer-load requirement of the
built-in LB/VFD workflow.

Use another pricing backend when ignored diagonal mass makes QMETIS a poor
search heuristic.

Failure and fallback behavior
-----------------------------

An incompatible or missing native library raises ``ImportError`` when QMETIS
is selected, not when Asunder itself is imported. Invalid weights, part counts,
or balance inputs raise ``ValueError``. The load-balancing pricing adapter
falls back to positive original topology or a deterministic balanced candidate
when the clipped pricing graph contains no useful positive edge.

See :doc:`../getting_started/quickstart` for the ordinary load-balancing
workflow and :doc:`../api/load_balancing/algorithms/qmetis` for complete API
signatures.
