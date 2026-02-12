# simulated_annealing_steiner.py
"""
Simulated Annealing Algorithm for Steiner Tree - Core Functions
MODIFIED: Uses EXHAUSTIVE cost function instead of ACC+AOC
"""

import networkx as nx
import random
import copy
import math


def connect_constrained_nodes(graph, solution_graph, power_nodes, constrained_nodes, weight_attr='weight'):
    """Connect constrained nodes to nearest power node"""
    for constrained_node in constrained_nodes:
        minimum_weight = 10000
        best_power_node = None

        for power_node in power_nodes:
            edge_weight = graph[constrained_node][power_node][weight_attr]

            if edge_weight < minimum_weight:
                best_power_node = power_node
                minimum_weight = edge_weight
                type_of_power_node = graph.nodes[best_power_node]["node_type"]

        solution_graph.add_node(constrained_node, node_type="constrained",
                               capacity=graph.nodes[constrained_node].get("capacity", 1), links=1)

        if best_power_node not in solution_graph.nodes():
            solution_graph.add_node(best_power_node, node_type=type_of_power_node,
                                   capacity=graph.nodes[best_power_node]["capacity"], links=0)

        solution_graph.nodes[best_power_node]["links"] = solution_graph.nodes[best_power_node]["links"] + 1
        solution_graph.add_edge(constrained_node, best_power_node, **{weight_attr: minimum_weight})
        solution_graph.nodes[constrained_node]["power_node"] = best_power_node

    return solution_graph


def connect_power_nodes(graph, initial_solution_graph, power_nodes, weight_attr='weight'):
    """Connect power nodes using Kruskal-like algorithm"""
    processed_nodes = list()
    edges_list = list()
    sorted_edges_list = list()

    for first_power_node in power_nodes:
        for second_power_node in power_nodes:
            if first_power_node != second_power_node:
                if second_power_node not in processed_nodes:
                    edge_weight = graph[first_power_node][second_power_node][weight_attr]
                    edges_list.append((first_power_node, second_power_node, edge_weight))
        processed_nodes.append(first_power_node)

    edges_list.sort(key=lambda a: a[2])

    for edges in edges_list:
        first_power_node = edges[0]
        second_power_node = edges[1]
        edge_weight = edges[2]

        if first_power_node not in initial_solution_graph.nodes():
            type_of_power_node = graph.nodes[first_power_node]["node_type"]
            initial_solution_graph.add_node(first_power_node, node_type=type_of_power_node,
                                          capacity=graph.nodes[first_power_node]["capacity"], links=0)
        if second_power_node not in initial_solution_graph.nodes():
            type_of_power_node = graph.nodes[second_power_node]["node_type"]
            initial_solution_graph.add_node(second_power_node, node_type=type_of_power_node,
                                          capacity=graph.nodes[second_power_node]["capacity"], links=0)

        initial_solution_graph.add_edge(first_power_node, second_power_node, **{weight_attr: edge_weight})

        # Check for cycles in undirected graph
        try:
            initial_solution_graph.remove_edge(first_power_node, second_power_node)
            if nx.has_path(initial_solution_graph, first_power_node, second_power_node):
                pass
            else:
                initial_solution_graph.add_edge(first_power_node, second_power_node, **{weight_attr: edge_weight})
                initial_solution_graph.nodes[first_power_node]["links"] = initial_solution_graph.nodes[first_power_node]["links"] + 1
                initial_solution_graph.nodes[second_power_node]["links"] = initial_solution_graph.nodes[second_power_node]["links"] + 1

                if first_power_node < second_power_node:
                    sorted_edges_list.append((first_power_node, second_power_node))
                else:
                    sorted_edges_list.append((second_power_node, first_power_node))
        except:
            initial_solution_graph.add_edge(first_power_node, second_power_node, **{weight_attr: edge_weight})
            initial_solution_graph.nodes[first_power_node]["links"] = initial_solution_graph.nodes[first_power_node]["links"] + 1
            initial_solution_graph.nodes[second_power_node]["links"] = initial_solution_graph.nodes[second_power_node]["links"] + 1

            if first_power_node < second_power_node:
                sorted_edges_list.append((first_power_node, second_power_node))
            else:
                sorted_edges_list.append((second_power_node, first_power_node))

    return initial_solution_graph, sorted_edges_list


def generate_initial_solution(graph, constrained_nodes, power_nodes, weight_attr='weight'):
    """
    Generate initial solution
    MODIFIED: Ensures initial solution respects discretionary constraints
    """
    initial_solution_graph = nx.Graph()
    
    # Step 1: Connect constrained (weak) nodes to power nodes
    initial_solution_graph = connect_constrained_nodes(
        graph, initial_solution_graph, power_nodes, constrained_nodes, weight_attr
    )
    
    # Step 2: Connect power nodes together
    initial_solution_graph, power_nodes_edges = connect_power_nodes(
        graph, initial_solution_graph, power_nodes, weight_attr
    )
    
    # Step 3: VALIDATE - Remove useless discretionary nodes
    # A discretionary is useless if it only connects to other discretionary
    nodes_to_remove = []
    
    for node in initial_solution_graph.nodes():
        node_type = initial_solution_graph.nodes[node].get('node_type')
        
        if node_type == 'power_discretionary':
            neighbors = list(initial_solution_graph.neighbors(node))
            
            # Check if connects to at least one weak or mandatory
            has_useful_connection = False
            for neighbor in neighbors:
                neighbor_type = initial_solution_graph.nodes[neighbor].get('node_type')
                if neighbor_type in ['constrained', 'power_mandatory']:
                    has_useful_connection = True
                    break
            
            # Mark for removal if useless
            if not has_useful_connection:
                nodes_to_remove.append(node)
    
    # Remove useless discretionary nodes
    if nodes_to_remove:
        for node in nodes_to_remove:
            # Before removing, reconnect any constrained nodes that were connected to it
            neighbors = list(initial_solution_graph.neighbors(node))
            constrained_neighbors = [
                n for n in neighbors 
                if initial_solution_graph.nodes[n].get('node_type') == 'constrained'
            ]
            
            # Reconnect constrained nodes to their nearest power node
            for constrained_node in constrained_neighbors:
                # Find nearest power node (excluding the one we're removing)
                min_weight = float('inf')
                best_power = None
                
                for power_node in power_nodes:
                    if power_node == node:
                        continue
                    if graph.has_edge(constrained_node, power_node):
                        weight = graph[constrained_node][power_node][weight_attr]
                        if weight < min_weight:
                            min_weight = weight
                            best_power = power_node
                
                if best_power:
                    # Remove old connection
                    if initial_solution_graph.has_edge(constrained_node, node):
                        initial_solution_graph.remove_edge(constrained_node, node)
                    
                    # Add new connection
                    if not initial_solution_graph.has_node(best_power):
                        initial_solution_graph.add_node(
                            best_power, 
                            node_type=graph.nodes[best_power]['node_type'],
                            capacity=graph.nodes[best_power]['capacity'],
                            links=0
                        )
                    
                    initial_solution_graph.add_edge(constrained_node, best_power, **{weight_attr: min_weight})
                    initial_solution_graph.nodes[best_power]["links"] += 1
            
            # Now remove the useless discretionary node
            initial_solution_graph.remove_node(node)
            power_nodes.remove(node)
        
        # Rebuild power_nodes_edges after removal
        initial_solution_graph, power_nodes_edges = connect_power_nodes(
            graph, initial_solution_graph, power_nodes, weight_attr
        )
    
    return initial_solution_graph, power_nodes_edges

def validate_discretionary_nodes(tree):
    """
    Validate that discretionary nodes are useful (not isolated)
    
    A discretionary node is valid if it connects at least one:
    - Constrained (weak) node, OR
    - Power mandatory node
    
    If a discretionary only connects to other discretionary nodes,
    it's useless and the solution is invalid.
    
    Returns:
        True if all discretionary nodes are useful, False otherwise
    """
    for node in tree.nodes():
        node_type = tree.nodes[node].get('node_type')
        
        if node_type == 'power_discretionary':
            neighbors = list(tree.neighbors(node))
            
            # Check if at least one neighbor is weak or mandatory
            has_useful_connection = False
            for neighbor in neighbors:
                neighbor_type = tree.nodes[neighbor].get('node_type')
                if neighbor_type in ['constrained', 'power_mandatory']:
                    has_useful_connection = True
                    break
            
            # If discretionary only connects to other discretionary, it's invalid
            if not has_useful_connection:
                return False
    
    return True


def calculate_exhaustive_cost(tree, network_graph, weight_attr='weight'):
    """
    MODIFIED: Calculate cost using EXHAUSTIVE formula instead of ACC+AOC
    
    ADDED: Validates that discretionary nodes are useful (not isolated)
    
    This replaces calculate_acc() + calculate_aoc() with the same cost function
    used by the exhaustive algorithm for fair comparison.
    
    Args:
        tree: Current solution tree
        network_graph: Original complete graph (for global_max_weight)
        weight_attr: Edge weight attribute name
        
    Returns:
        total_cost, edge_cost, degree_cost
    """
    if tree.number_of_edges() == 0:
        return float('inf'), float('inf'), float('inf')
    
    # CRITICAL: Validate topology before calculating cost
    if not validate_discretionary_nodes(tree):
        # Invalid topology - discretionary node is isolated/useless
        return float('inf'), float('inf'), float('inf')
    
    # Get node types
    weak_nodes = [n for n in tree.nodes() if tree.nodes[n].get('node_type') == 'constrained']
    mandatory_nodes = [n for n in tree.nodes() if tree.nodes[n].get('node_type') == 'power_mandatory']
    discretionary_nodes = [n for n in tree.nodes() if tree.nodes[n].get('node_type') == 'power_discretionary']
    
    max_nodes = len(weak_nodes) + len(mandatory_nodes) + len(discretionary_nodes)
    max_possible_edges = max_nodes - 1
    
    # CRITICAL: Get global max weight from ORIGINAL GRAPH, not solution tree
    global_max_weight = max(network_graph[u][v][weight_attr] for u, v in network_graph.edges())
    
    # ===== EDGE COST =====
    total_weight = sum(tree[u][v][weight_attr] for u, v in tree.edges())
    max_possible_weight = global_max_weight * max_possible_edges
    edge_cost = total_weight / max_possible_weight if max_possible_weight > 0 else 0
    
    # ===== DEGREE COST =====
    all_power = set(mandatory_nodes + discretionary_nodes)
    power_in_tree = [n for n in tree.nodes() if n in all_power]
    
    degree_sum = 0
    for node in power_in_tree:
        capacity = tree.nodes[node].get('capacity', 1)
        if capacity > 0 and capacity != float('inf'):
            degree_sum += tree.degree(node) / capacity
    
    degree_cost_raw = degree_sum / len(power_in_tree) if power_in_tree else 0
    
    # Normalize to [0,1]
    all_power_with_capacity = [n for n in all_power if tree.nodes[n].get('capacity', 1) != float('inf')]
    if all_power_with_capacity:
        min_capacity = min(tree.nodes[n].get('capacity', 1) for n in all_power_with_capacity)
    else:
        min_capacity = 1
    
    max_degree_possible = max_nodes - 1
    max_degree_cost_per_node = max_degree_possible / min_capacity if min_capacity > 0 else 1
    degree_cost = degree_cost_raw / max_degree_cost_per_node if max_degree_cost_per_node > 0 else 0
    
    # ===== TOTAL COST =====
    total_cost = edge_cost + degree_cost
    
    return total_cost, edge_cost, degree_cost


# Keep old functions for backward compatibility (they just call new function)
def calculate_aoc(graph):
    """DEPRECATED: Kept for compatibility, not used anymore"""
    nodes_info = graph.nodes.data()
    cost = 0
    for node in graph.nodes():
        quantity_of_links = len(list(graph.neighbors(node)))
        cost = cost + quantity_of_links * graph.nodes[node]["capacity"]
    aoc = 2 * (cost / len(nodes_info))
    return aoc


def calculate_acc(graph, weight_attr='weight'):
    """DEPRECATED: Kept for compatibility, not used anymore"""
    average_weight = nx.average_shortest_path_length(graph, weight=weight_attr)
    max_weight = 1
    acc = (average_weight / max_weight) / 50
    return acc


def eliminate_discretionary_power_node(network_graph, graph, power_nodes, discretionary_power_nodes,
                                     power_nodes_edges, eliminated_discretionary_power_nodes, weight_attr='weight'):
    """Eliminate a random discretionary power node"""
    power_nodes_copy = power_nodes[:]
    discretionary_power_nodes_copy = discretionary_power_nodes[:]
    power_nodes_edges_copy = power_nodes_edges[:]
    eliminated_discretionary_power_nodes_copy = eliminated_discretionary_power_nodes[:]

    if len(discretionary_power_nodes) > 0:
        selected_discretionary_power_node = random.choice(discretionary_power_nodes)
    else:
        return graph, power_nodes, discretionary_power_nodes, power_nodes_edges, eliminated_discretionary_power_nodes

    linked_power_nodes = list()
    linked_constrained_nodes = list()
    neighbors = list(graph.neighbors(selected_discretionary_power_node))

    for neighbor in neighbors:
        type_of_node = graph.nodes[neighbor]["node_type"]
        graph.nodes[neighbor]["links"] = graph.nodes[neighbor]["links"] - 1
        if type_of_node == "constrained":
            linked_constrained_nodes.append(neighbor)
        else:
            linked_power_nodes.append(neighbor)

    quantity_of_constrained_nodes = len(linked_constrained_nodes)
    quantity_of_power_nodes = len(linked_power_nodes)

    if quantity_of_power_nodes < 1 and quantity_of_constrained_nodes < 1:
        return graph, power_nodes, discretionary_power_nodes, power_nodes_edges, eliminated_discretionary_power_nodes

    for power_node in power_nodes:
        graph.nodes[power_node]["links"] = 0
    for discretionary_node in discretionary_power_nodes:
        graph.nodes[discretionary_node]["links"] = 0

    graph.remove_edges_from(power_nodes_edges)
    graph.nodes[selected_discretionary_power_node]["links"] = 0
    graph.remove_node(selected_discretionary_power_node)
    power_nodes_copy.remove(selected_discretionary_power_node)
    eliminated_discretionary_power_nodes_copy.append(selected_discretionary_power_node)
    discretionary_power_nodes_copy.remove(selected_discretionary_power_node)
    graph = connect_constrained_nodes(network_graph, graph, power_nodes_copy, linked_constrained_nodes, weight_attr)
    graph, power_nodes_edges_copy = connect_power_nodes(network_graph, graph, power_nodes_copy, weight_attr)

    return graph, power_nodes_copy, discretionary_power_nodes_copy, power_nodes_edges_copy, eliminated_discretionary_power_nodes_copy


def change_reference_power_node(network_graph, graph, power_nodes, power_nodes_edges, discretionary_power_nodes,
                               eliminated_discretionary_power_nodes, constrained_nodes, weight_attr='weight'):
    """Change reference power node for some constrained nodes"""
    number_of_constrained_nodes_to_edit = min(2, len(constrained_nodes))

    constrained_nodes_to_edit = list()
    selected_constrained = 0

    while selected_constrained < number_of_constrained_nodes_to_edit:
        constrained_node = random.choice(list(constrained_nodes))
        if constrained_node not in constrained_nodes_to_edit:
            constrained_nodes_to_edit.append(constrained_node)
            selected_constrained = selected_constrained + 1

    for constrained in constrained_nodes_to_edit:
        previous_power_node = graph.nodes[constrained]["power_node"]
        graph.remove_edge(constrained, previous_power_node)
        graph.nodes[previous_power_node]["links"] = graph.nodes[previous_power_node]["links"] - 1

        choice = random.randrange(2)
        if choice == 0 and len(eliminated_discretionary_power_nodes) > 0:
            random_power_node = random.choice(eliminated_discretionary_power_nodes)
            eliminated_discretionary_power_nodes.remove(random_power_node)
            power_nodes.append(random_power_node)
            discretionary_power_nodes.append(random_power_node)
            graph, power_nodes_edges = connect_power_nodes(network_graph, graph, power_nodes, weight_attr)
        else:
            random_power_node = random.choice(power_nodes)

        edge_weight = network_graph[constrained][random_power_node][weight_attr]
        graph.add_edge(constrained, random_power_node, **{weight_attr: edge_weight})
        graph.nodes[random_power_node]["links"] = graph.nodes[random_power_node]["links"] + 1
        graph.nodes[constrained]["power_node"] = random_power_node

    return graph, power_nodes, power_nodes_edges, discretionary_power_nodes, eliminated_discretionary_power_nodes


def simulated_annealing(network_graph, initial_solution_graph, constrained_nodes, power_nodes,
                       discretionary_power_nodes, power_nodes_edges,
                       initial_temperature=100, minimum_temperature=0.0001, k=1, weight_attr='weight'):
    """
    Main simulated annealing algorithm
    MODIFIED: Uses calculate_exhaustive_cost() instead of ACC+AOC
    """
    sa_solution_graph = initial_solution_graph
    temperature = initial_temperature
    eliminated_discretionary_power_nodes = []
    
    # MODIFIED: Use exhaustive cost function
    solution_cost, _, _ = calculate_exhaustive_cost(sa_solution_graph, network_graph, weight_attr)

    temp_power_nodes_edges = power_nodes_edges
    temp_power_nodes = power_nodes[:]
    temp_discretionary_power_nodes = discretionary_power_nodes[:]

    number_of_iterations = 0

    while temperature > minimum_temperature:
        number_of_iterations = number_of_iterations + 1

        sa_solution_graph_copy = copy.deepcopy(sa_solution_graph)
        power_nodes_copy = power_nodes[:]
        power_nodes_edges_copy = power_nodes_edges[:]
        discretionary_power_nodes_copy = discretionary_power_nodes[:]
        eliminated_discretionary_power_nodes_copy = eliminated_discretionary_power_nodes[:]

        move = random.randrange(2)

        if move == 0:
            temporary_graph, temp_power_nodes, temp_discretionary_power_nodes, temp_power_nodes_edges, temp_eliminated_discretionary_power_nodes = eliminate_discretionary_power_node(
                network_graph, sa_solution_graph_copy, power_nodes_copy, discretionary_power_nodes_copy,
                power_nodes_edges_copy, eliminated_discretionary_power_nodes_copy, weight_attr)
        elif move == 1:
            temporary_graph, temp_power_nodes, temp_power_nodes_edges, temp_discretionary_power_nodes, temp_eliminated_discretionary_power_nodes = change_reference_power_node(
                network_graph, sa_solution_graph_copy, power_nodes_copy, power_nodes_edges_copy,
                discretionary_power_nodes_copy, eliminated_discretionary_power_nodes_copy, constrained_nodes, weight_attr)

        # MODIFIED: Use exhaustive cost function
        iteration_cost, _, _ = calculate_exhaustive_cost(temporary_graph, network_graph, weight_attr)
        delta_energy = iteration_cost - solution_cost

        if delta_energy < 0:
            sa_solution_graph = temporary_graph
            solution_cost = iteration_cost
            power_nodes_edges = temp_power_nodes_edges
            power_nodes = temp_power_nodes
            discretionary_power_nodes = temp_discretionary_power_nodes
            eliminated_discretionary_power_nodes = temp_eliminated_discretionary_power_nodes
        else:
            random_factor = random.random()
            if random_factor < math.exp((k * -delta_energy) / (temperature)):
                sa_solution_graph = temporary_graph
                solution_cost = iteration_cost
                power_nodes_edges = temp_power_nodes_edges
                power_nodes = temp_power_nodes
                discretionary_power_nodes = temp_discretionary_power_nodes
                eliminated_discretionary_power_nodes = temp_eliminated_discretionary_power_nodes

        temperature = temperature * 0.99

    return sa_solution_graph




def run_sa_algorithm(network_graph, constrained_nodes, mandatory_power_nodes, discretionary_power_nodes,
                    initial_temperature=100, k_factor=12, weight_attr='weight'):
    """
    High-level function to run complete SA algorithm
    MODIFIED: Fallback to mandatory-only if final solution is invalid
    """
    power_nodes = list(mandatory_power_nodes) + list(discretionary_power_nodes)
    total_nodes_quantity = len(constrained_nodes) + len(mandatory_power_nodes) + len(discretionary_power_nodes)

    # Generate initial solution WITH all discretionary
    initial_solution_graph, power_nodes_edges = generate_initial_solution(
        network_graph, constrained_nodes, power_nodes, weight_attr)

    initial_cost, _, _ = calculate_exhaustive_cost(initial_solution_graph, network_graph, weight_attr)
    
    try:
        initial_average_weight = nx.average_shortest_path_length(initial_solution_graph, weight=weight_attr)
    except:
        initial_average_weight = float('inf')

    # Run SA
    sa_solution = simulated_annealing(
        network_graph, initial_solution_graph, constrained_nodes, power_nodes,
        discretionary_power_nodes, power_nodes_edges,
        initial_temperature=initial_temperature, k=k_factor * total_nodes_quantity, weight_attr=weight_attr)

    # Check if final solution is valid
    final_cost, _, _ = calculate_exhaustive_cost(sa_solution, network_graph, weight_attr)
    
    # If invalid, fallback to mandatory-only solution
    if final_cost == float('inf'):
        # Build fallback solution with only mandatory
        fallback_power = list(mandatory_power_nodes)
        fallback_solution, fallback_edges = generate_initial_solution(
            network_graph, constrained_nodes, fallback_power, weight_attr
        )
        
        fallback_cost, _, _ = calculate_exhaustive_cost(fallback_solution, network_graph, weight_attr)
        
        # Use fallback
        sa_solution = fallback_solution
        final_cost = fallback_cost
    
    try:
        final_average_weight = nx.average_shortest_path_length(sa_solution, weight=weight_attr)
    except:
        final_average_weight = float('inf')

    return sa_solution, initial_cost, final_cost, initial_average_weight, final_average_weight
