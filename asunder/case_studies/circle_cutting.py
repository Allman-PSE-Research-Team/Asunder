"""Circle cutting case-study graph builder."""

import networkx as nx


def build_circle_cutting_graph(num_circles, num_rectangles, dimensions):
    """Build a circle-cutting constraint graph used in benchmark experiments.

    Args:
        num_circles: Number of circles in the instance.
        num_rectangles: Number of candidate rectangles.
        dimensions: Coordinate dimensions used in non-overlap and boundary terms.

    Returns:
        Tuple of ``(graph, constraint_labels, var_to_constraints)``.
    """
    constraint_id = 0
    G = nx.Graph()

    # Constraint abbreviations
    constraint_labels = {
        'CA': 'Circle Assignment',
        'A': 'Area',
        'NO': 'Non-overlap',
        'BP': 'Boundary Position',
        'OBJ': 'Objective Coupling'
    }

    # Function to add constraint nodes
    def add_constraint(constraint_type, details):
        nonlocal constraint_id
        constraint_id += 1
        node_label = f'{constraint_type}{constraint_id}'
        G.add_node(node_label, type='constraint', constraint=constraint_type, details=details)
        return node_label

    # Store variables for linking constraints
    var_to_constraints = {}
    def link_constraints(var, constraint_node):
        # var_type = 'integer' if var.startswith('α') else 'continuous'
        if var.startswith('α'):
            var_type = 'integer'
        elif var.startswith('y'):
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

    # Constraint (2): Circle assignment
    for j in range(1, num_circles + 1):
        vars_in_constraint = [f'α{j}{r}' for r in range(1, num_rectangles + 1)]
        node = add_constraint('CA', {'circle': j})
        for var in vars_in_constraint:
            link_constraints(var, node)

    # Constraints per rectangle
    for r in range(1, num_rectangles + 1):
        # Add y_r variable (rectangle activation)
        # var_to_constraints[f'y{r}'] = []

        # # Constraint (13): Area
        # # + help cluster the subgraph for a given rectangle since all alpha_jr are tied together through this node
        # # + help tie the constraint involving sa set of circles
        # # + makes the CA nodes less structurally isolated
        # # - NO and BP constraints already sufficiently capturee rectangle-local structure
        # # - Might cause "dominating"cliques
        # # In general, not strictly necessary but is usually helpful in: emphasizing rectangle-specific coupling and reinforcing structural symmetry between CA and rectangle-level constraints

        # area_vars = [f'α{j}{r}' for j in range(1, num_circles + 1)]
        # area_node = add_constraint('A', {'rectangle': r})
        # for var in area_vars:
        #     link_constraints(var, area_node)

        # Constraint (15): Non-overlap
        for i in range(1, num_circles + 1):
            for j in range(i + 1, num_circles + 1):
                non_overlap_vars = [f'α{i}{r}', f'α{j}{r}'] + \
                                   [f'x{i}{d}{r}' for d in dimensions] + \
                                   [f'x{j}{d}{r}' for d in dimensions]
                no_node = add_constraint('NO', {'rectangle': r, 'circles': (i, j)})
                for var in non_overlap_vars:
                    link_constraints(var, no_node)

        # Constraints (7) & (8): Boundary position
        for j in range(1, num_circles + 1):
            boundary_vars = [f'α{j}{r}'] + [f'x{j}{d}{r}' for d in dimensions]
            bp_node = add_constraint('BP', {'rectangle': r, 'circle': j})
            for var in boundary_vars:
                link_constraints(var, bp_node)

        # # Constraint: objective coupling via q_r = π y_r ∑_j α_{jr} R_j^2
        # y_var = f'y{r}'  # new continuous variable for rectangle r
        # obj_vars = [y_var] + [f'α{j}{r}' for j in range(1, num_circles + 1)]
        # obj_node = add_constraint('OBJ', {'rectangle': r})

        # for var in obj_vars:
        #     link_constraints(var, obj_node)

    return G, constraint_labels, var_to_constraints
