"""cpcong case-study graph builder."""

import networkx as nx


def build_cpcong_graph(K, J, T):
    """Build a CP-constrained unit-commitment style constraint graph.

    Args:
        K: Number of production units.
        J: Number of products.
        T: Number of time periods.

    Returns:
        Tuple of ``(graph, constraint_labels, var_to_constraints)``.
    """
    constraint_id = 0
    G = nx.Graph()

    # Constraint abbreviations
    constraint_labels = {
        'DS': 'Demand Satisfaction',
        'UAC': 'Unit Activation Coupling',
        'L': 'Lead Time (Nonlinear)',
        'CB': 'Cummulative Build'
    }
    integer_variables = ['y', 'z']

    # Function to add constraint nodes
    def add_constraint(constraint_type, details=None):
        nonlocal constraint_id
        constraint_id += 1
        node_label = f'${constraint_type}_' + '{' + str(constraint_id) + '}$'
        G.add_node(node_label, type='constraint', constraint=constraint_type, details=details)
        return node_label

    # Store variables for linking constraints
    var_to_constraints = {}
    def link_constraints(var, constraint_node):
        if any(var.startswith(i) for i in integer_variables):
            var_type = 'integer'
        else:
            var_type = 'continuous'

        if var not in var_to_constraints:
            var_to_constraints[var] = []

        for existing_constraint in var_to_constraints[var]:
            if G.has_edge(existing_constraint, constraint_node):
                edge = G[existing_constraint][constraint_node]
                edge['weight'] += 1
                edge['variables'].append(var)
                edge['var_types'].add(var_type)
            else:
                G.add_edge(
                    existing_constraint, constraint_node,
                    weight=1,
                    variables=[var],
                    var_types={var_type}
                )

            # ✅ Set final var_type to continuous if mixed
            edge = G[existing_constraint][constraint_node]
            if edge['var_types'] == {'integer'}:
                edge['var_type'] = 'integer'
            else:
                edge['var_type'] = 'continuous'

        var_to_constraints[var].append(constraint_node)

    # Add constraints

    # Constraint (32): Demand Satisfaction
    for t in range(1, T + 1):
        for j in range(1, J + 1):
            vars_in_constraint = [f'x{k}{j}{t}' for k in range(1, K + 1)]
            node = add_constraint('DS', {'product': j, 'time': t})
            for var in vars_in_constraint:
                link_constraints(var, node)

    # Constraint (33): Unit Activation Coupling
    for t in range(1, T + 1):
        for j in range(1, J + 1):
            for k in range(1, K + 1):
                vars_in_constraint = [f'x{k}{j}{t}', f"y{k}{t}"]
                node = add_constraint('UAC', {'production_unit': k, 'product': j, 'time': t})
                for var in vars_in_constraint:
                    link_constraints(var, node)

    # Constraint (36): Lead Time (Nonlinear)
    for t in range(1, T + 1):
        for k in range(1, K + 1):
            vars_in_constraint = [f'x{k}{j}{t}' for j in range(1, J + 1)]
            node = add_constraint('L', {'production_unit': k, 'time': t})
            for var in vars_in_constraint:
                link_constraints(var, node)

    # Constraint (23): Cummulative Build
    for t in range(1, T + 1):
        for k in range(1, K + 1):
            vars_in_constraint = [f"y{k}{t}"] + [f'z{k}{t_}' for t_ in range(1, t + 1)]
            node = add_constraint('CU', {'production_unit': k, 'time': t})
            for var in vars_in_constraint:
                link_constraints(var, node)

    # OPTIONAL
    # # Constraint: objective coupling ?

    return G, constraint_labels, var_to_constraints
