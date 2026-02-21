Problem Fit and Utility
=======================

Asunder is most useful for optimization problems where constraints interact in a structured way and those interactions can be modeled as a graph. In these settings, column-generation with graph-based partitioning often finds high-value decompositions without heavy customization.

In addition to the domain patterns below, Asunder supports general constrained partitioning whenever key requirements can be encoded as must-link and cannot-link constraints.

Constraint Graph Compatibility
-----------------------------

For built-in case-study evaluation flows (``run_evaluation``), Asunder expects a constraint-graph schema similar to the packaged case studies.

Required fields
^^^^^^^^^^^^^^^

- graph type: undirected graph (typically ``networkx.Graph``)
- node attribute: ``constraint`` (string tag)
- edge attribute: ``var_type`` with values ``"integer"`` or ``"continuous"``

Recommended fields
^^^^^^^^^^^^^^^^^^

- node attributes: ``type`` (for example ``"constraint"``), ``details`` (metadata)
- edge attributes: ``weight``, ``variables``, ``var_types``

Operational meaning in built-in workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``constraint`` is used to define ground-truth role labels and to identify nonlinear/core groups
- ``var_type`` is used to derive edge subsets used by CP and CD_Refine evaluation paths

If you use decomposition APIs directly (instead of ``run_evaluation``), the strict case-study graph schema is not required. You can provide adjacency data and constraints directly through ``must_link``, ``cannot_link``, and ``worthy_edges``.

Where the default approach works well
-------------------------------------

The default pipeline is usually a good first choice when:

- must-link, cannot-link, or worthy-edge semantics are meaningful in the model
- coordination or operations are coupled across time
- constraints can be mapped to nodes and shared-variable interactions to edges
- the problem has mixed discrete-continuous behavior with local clusters and global coupling
- you want interpretable groups (for example modules, teams, corridors, network regions)

Domain examples
---------------

Stochastic Design and Dispatch in Energy Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- unit commitment, expansion, and dispatch decisions often couple across time, scenarios, and network constraints
- graph structure commonly reflects temporal continuity, transmission coupling, and shared resource constraints
- Asunder can separate strongly coordinated operational blocks while preserving cross-block interactions through the master/pricing loop

Scheduling and Resource Allocation in Healthcare Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- staffing, bed allocation, OR scheduling, and patient-flow constraints are highly coupled over shifts and days
- constraint graphs often contain repeated motifs (wards, teams, service lines) with inter-unit coupling
- Asunder can reveal decomposition blocks aligned with care pathways and resource coordination patterns

Planning, Routing, and Location in Supply Chain and Logistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- facility location, routing, and inventory/replenishment decisions share temporal and capacity coupling
- graph views naturally capture shared assets, lane constraints, and demand-synchronization structure
- Asunder can improve tractability by isolating dense local coordination while coordinating global tradeoffs in the master

Network Configuration and Resource Management in Telecommunications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- spectrum allocation, routing, placement, and resilience constraints couple across time and topology
- interaction graphs typically have region-like communities with high local density and sparse long-range coupling
- Asunder can identify operationally meaningful partitions for staged optimization

When to customize
-----------------

Customization is usually helpful when:

- your preferred objective surrogate differs from modularity-style scoring
- domain constraints require specialized feasible-column generators
- pricing needs domain-specific heuristics
- post-processing/refinement rules are strict and problem-specific

See ``custom_methods.rst`` for integration contracts.
