Weighted Load Balancing
========================

Start with :doc:`../getting_started/quickstart` for a complete ``LoadBalancer``
example. This page explains the units behind its balance settings.

Counts versus loads
-------------------

With unit node weights, balancing means balancing node counts. With supplied
weights, each community's load is the sum of its members' weights. Contracting
must-linked nodes sums their weights, so total load does not change.

``LoadBalancer``, both VFD implementations, and QMETIS require positive integer
node weights. ModularVFD retains this requirement even when
``use_K_constraint=False``. Fractional node weights are rejected, not silently
rounded or scaled.

The generic ``run_csd_decomposition`` interface still accepts finite real
``node_weights`` for custom hooks. A hook's own restrictions apply when it
consumes those weights. Additional application-specific vectors can be passed
under distinct hook argument names.

The K/R rule
-------------

Let :math:`W=\sum_i w_i` be total node weight. ``K`` is the requested number of
communities; ``R`` is an absolute load-window width in the same units as the
weights, not a percentage. Without explicit ``R_bounds``, the rule is:

.. math::

   L = \max\left(1,\left\lfloor \frac{W}{K}-\frac{R}{2}+\frac12\right\rfloor\right),
   \qquad U=L+R.

Each community's load must lie between ``L`` and ``U``, inclusive. The shared
bound resolver also caps ``U`` at ``W``. For example, ``W=120, K=4, R=10``
gives the interval ``[25, 35]``. With unit weights, ``W`` is simply the node
count and the original size-balancing rule is recovered.

``R_bounds=(lower, upper)`` replaces the derived interval. Its endpoints use
the same integer units as node weights. ``R=0`` without explicit bounds asks
for equal loads; divisibility of ``W`` by ``K`` is necessary but does not
guarantee feasibility because nodes and must-link components are indivisible.

Choosing integer units
----------------------

Choose a unit that represents the precision your application needs, such as
milliseconds instead of seconds. Supply the converted positive integer
weights yourself; Asunder does not choose a scale or precision for you.

For example, loads ``[0.25, 0.75, 1.0]`` with bounds ``[0.75, 1.25]`` can be
expressed in hundredths as weights ``[25, 75, 100]`` and
``R_bounds=(75, 125)``. Keep ``K`` unchanged. Convert any custom load thresholds
to the same units, and divide reported community loads by 100 to interpret
them in the original units. Do not scale the adjacency or modularity
resolution merely because node-load units changed.

Choose a scale that represents the intended weights and bounds exactly.
Rounding arbitrary values is an application-level approximation that can
change hard feasibility. Avoid unnecessarily large units that risk overflow
or loss of numerical precision.

Why explicit bounds matter when rescaling
------------------------------------------

The rounded K/R rule is not invariant under a change of units:

- ``W=10, K=3, R=1`` gives ``[3, 4]``.
- Scaling weights and ``R`` by 10 gives ``W=100, K=3, R=10`` and ``[28, 38]``,
  not ``[30, 40]``.

To preserve the original feasible load interval, pass the explicitly scaled
``R_bounds=(30, 40)``. Scaling ``R`` alone is not enough to preserve rounded
bounds. This also applies to custom constraints that interpret node weights.

QMETIS edge quantization is different
--------------------------------------

The :doc:`qmetis` adapter can quantize fractional **edge** weights to guide its
candidate search, then recompute reduced costs with the original data.
That mechanism does not scale node weights or balance bounds. Node weights
affect hard feasibility, which exact objective rescoring cannot restore after
inappropriate rounding.
