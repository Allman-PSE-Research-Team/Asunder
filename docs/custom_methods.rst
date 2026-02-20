Custom Method Contracts
=======================

Master problem callable
-----------------------

The custom master callable must accept:

- ``A, a, m, Z_star, f_stars``
- keyword constraints (for example ``must_link``, ``cannot_link``)

and return either:

- ``(lambda_sol, master_obj_val)`` for integer master mode, or
- ``(lambda_sol, duals, master_obj_val)`` for dual extraction mode.

Subproblem callable
-------------------

The custom subproblem callable must accept:

- ``A, a, m, duals``

and return:

- ``(sub_obj_val, z_sol)``, where ``z_sol`` is a partition/co-association matrix.

