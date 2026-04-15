Problem Fit and Utility
=======================

Asunder is most useful when a larger optimization or partitioning problem has a
meaningful graph representation and the resulting grouping decisions are useful
for decomposition, coordination analysis, or parallel computing.

In those settings, the package provides two complementary layers:

- ``asunder.base`` for reusable decomposition and partitioning tools
- ``asunder.nlbp`` for the built-in nonlinear branch-and-price application

Choosing Between ``base`` and ``nlbp``
--------------------------------------

Use ``asunder.base`` when:

- you already have a graph and want reusable decomposition primitives
- you want to plug in your own master, subproblem, or refinement logic
- your application is not the current nonlinear branch-and-price workflow
- you are building the next application package on top of the reusable core

Use ``asunder.nlbp`` when:

- you want the packaged nonlinear branch-and-price evaluation workflow which 
  imposes edge-based and cardinality constraints.
- you want the current built-in case-study graph builders
- you want the NLBP-specific refinement routine

What Good Problem Fit Looks Like
--------------------------------

The default approach is usually a good first choice when:

- must-link, cannot-link, or worthy-edge semantics are meaningful
- nodes represent constraints, tasks, assets, or entities with real
  coordination structure
- dense local interactions and sparser long-range interactions both matter
- a graph partition would make an optimization model easier to solve or easier
  to understand
- a heuristic pricing step is acceptable even if the final application still
  uses solver-backed components elsewhere

In practice, this often means the graph captures one or more of the following:

- coupling across time periods
- shared resources or shared decision variables
- mixed discrete-continuous interactions
- repeated motifs or region-like communities
- pairwise incompatibilities or enforced pairwise grouping decisions

What Poor Problem Fit Looks Like
--------------------------------

Asunder is usually a poor fit when:

- there is no meaningful graph representation of the problem
- the graph is almost uniform, with little structure to exploit
- the important constraints cannot be expressed through pairwise logic, do not 
  fit the Nonlinear Branch and Price workflow, and no custom extension is planned
- the application requires an exact pricing formulation and domain-specific
  logic, but you are not prepared to implement those custom pieces
- the partition itself is not operationally useful and only a full monolithic
  solve matters

Constraint Graph Compatibility
------------------------------

For the built-in case-study evaluation flows in ``run_evaluation``, Asunder
expects a constraint-graph schema similar to the packaged case studies in
``asunder.nlbp.case_studies``.

Required fields
^^^^^^^^^^^^^^^

- graph type: undirected graph, typically ``networkx.Graph``
- node attribute: ``constraint`` (string tag)
- edge attribute: ``var_type`` with values ``"integer"`` or ``"continuous"``

Recommended fields
^^^^^^^^^^^^^^^^^^

- node attributes: ``type`` and ``details``
- edge attributes: ``weight``, ``variables``, and ``var_types``

Operational meaning in built-in workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``constraint`` is used for ground-truth role labels and for identifying
  nonlinear/core groups
- ``var_type`` is used to derive the edge subsets needed by the CP and
  CD_Refine evaluation paths

If you use the reusable decomposition APIs directly instead of
``run_evaluation``, the strict case-study graph schema is not required. In that
mode you can work directly from adjacency data plus explicit
``must_link``, ``cannot_link``, and optional ``worthy_edges`` inputs.

Representative Domains
----------------------

Stochastic Design and Dispatch in Energy Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- unit commitment, expansion, and dispatch decisions often couple across time,
  scenarios, and network constraints
- graph structure naturally captures temporal continuity, transmission
  interactions, and shared operational constraints
- Asunder can separate dense operational blocks while preserving cross-block
  coordination through the master/pricing loop

Scheduling and Resource Allocation in Healthcare Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- staffing, bed allocation, OR scheduling, and patient-flow constraints are
  strongly coupled across shifts and days
- graph views often reveal repeated motifs such as wards, service lines, and
  resource teams
- Asunder can produce partitions aligned with practical coordination boundaries

Planning, Routing, and Location in Supply Chain and Logistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- facility location, routing, and inventory decisions share temporal and
  capacity coupling
- graph structure can encode shared assets, lane dependencies, and
  synchronization constraints
- Asunder can isolate dense local coordination while leaving broader tradeoffs
  to the master problem

Network Configuration and Resource Management in Telecommunications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- spectrum allocation, routing, placement, and resilience constraints often
  couple across time and topology
- interaction graphs frequently contain region-like clusters with sparse
  long-range coupling
- Asunder can help identify staged or hierarchical optimization structure

When To Customize
-----------------

Customization is usually helpful when:

- your preferred objective surrogate differs from modularity-style scoring
- initial feasible column generation is highly domain-specific
- pricing requires a custom heuristic or exact formulation
- the refinement step is application-specific
- the graph representation is only a starting point and needs extra domain
  logic to be operationally meaningful

As a rule of thumb:

- customize within ``asunder.base`` when the logic is reusable across
  applications
- extend ``asunder.nlbp`` or add a new peer application package when the logic
  is specific to one workflow or one family of case studies

See ``Reference -> Development Guide -> Special Topics`` for integration
contracts and extension points.
