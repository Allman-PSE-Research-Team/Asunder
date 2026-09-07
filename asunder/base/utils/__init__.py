"""Graph and partition utility exports used across Asunder."""

from asunder.base.utils.graph import (
    contract_adj_matrix_new,
    contract_node_pairs,
    contract_partition_matrix,
    expand_z_matrix,
    group_nodes_by_community,
    normalize_node_pairs,
    partition_matrix_to_vector,
    partition_satisfies_pairwise_constraints,
    partition_vector_to_2d_matrix,
    validate_partition_matrix,
)
from asunder.base.utils.partition_generation import (
    make_partitions_links_only,
    make_partitions_random_links_only,
    make_simple_partition,
)

__all__ = [
    "contract_adj_matrix_new",
    "contract_node_pairs",
    "contract_partition_matrix",
    "expand_z_matrix",
    "group_nodes_by_community",
    "make_partitions_links_only",
    "make_partitions_random_links_only",
    "make_simple_partition",
    "normalize_node_pairs",
    "partition_matrix_to_vector",
    "partition_satisfies_pairwise_constraints",
    "partition_vector_to_2d_matrix",
    "validate_partition_matrix",
]
