# simulated_annealing_steiner.py
"""
Simulated Annealing Algorithm for Steiner Tree - Core Functions
Adapted to work with externally provided graphs
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
    """Generate initial solution"""
    initial_solution_graph = nx.Graph()
    initial_solution_graph = connect_constrained_nodes(graph, initial_solution_graph, power_nodes, constrained_nodes, weight_attr)
    initial_solution_graph, power_nodes_edges = connect_power_nodes(graph, initial_solution_graph, power_nodes, weight_attr)
    return initial_solution_graph, power_nodes_edges


def calculate_aoc(graph):
    """Calculate Average Operational Cost"""
    nodes_info = graph.nodes.data()
    cost = 0

    for node in graph.nodes():
        quantity_of_links = len(list(graph.neighbors(node)))
        cost = cost + quantity_of_links * graph.nodes[node]["capacity"]

    aoc = 2 * (cost / len(nodes_info))
    return aoc


def calculate_acc(graph, weight_attr='weight'):
    """Calculate Average Communication Cost"""
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
    """Main simulated annealing algorithm"""
    sa_solution_graph = initial_solution_graph
    temperature = initial_temperature
    eliminated_discretionary_power_nodes = []
    solution_energy = calculate_acc(sa_solution_graph, weight_attr) + calculate_aoc(sa_solution_graph)

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

        iteration_energy = calculate_acc(temporary_graph, weight_attr) + calculate_aoc(temporary_graph)
        delta_energy = iteration_energy - solution_energy

        if delta_energy < 0:
            sa_solution_graph = temporary_graph
            solution_energy = iteration_energy
            power_nodes_edges = temp_power_nodes_edges
            power_nodes = temp_power_nodes
            discretionary_power_nodes = temp_discretionary_power_nodes
            eliminated_discretionary_power_nodes = temp_eliminated_discretionary_power_nodes
        else:
            random_factor = random.random()
            if random_factor < math.exp((k * -delta_energy) / (temperature)):
                sa_solution_graph = temporary_graph
                solution_energy = iteration_energy
                power_nodes_edges = temp_power_nodes_edges
                power_nodes = temp_power_nodes
                discretionary_power_nodes = temp_discretionary_power_nodes
                eliminated_discretionary_power_nodes = temp_eliminated_discretionary_power_nodes

        temperature = temperature * 0.99

    return sa_solution_graph


def run_sa_algorithm(network_graph, constrained_nodes, mandatory_power_nodes, discretionary_power_nodes,
                    initial_temperature=100, k_factor=12, weight_attr='weight'):
    """
    High-level function to run complete SA algorithm with provided graph

    Args:
        network_graph: NetworkX graph with all nodes and edges already created
        constrained_nodes: list of weak/constrained nodes
        mandatory_power_nodes: list of mandatory power nodes
        discretionary_power_nodes: list of discretionary power nodes
        initial_temperature: starting temperature for SA
        k_factor: multiplication factor for k parameter
        weight_attr: attribute name for edge weights ('weight' or 'latency')

    Returns:
        best_solution_graph, initial_cost, final_cost, initial_avg_weight, final_avg_weight
    """
    power_nodes = list(mandatory_power_nodes) + list(discretionary_power_nodes)
    total_nodes_quantity = len(constrained_nodes) + len(mandatory_power_nodes) + len(discretionary_power_nodes)

    initial_solution_graph, power_nodes_edges = generate_initial_solution(
        network_graph, constrained_nodes, power_nodes, weight_attr)

    initial_cost = calculate_acc(initial_solution_graph, weight_attr) + calculate_aoc(initial_solution_graph)
    initial_average_weight = nx.average_shortest_path_length(initial_solution_graph, weight=weight_attr)

    sa_solution = simulated_annealing(
        network_graph, initial_solution_graph, constrained_nodes, power_nodes,
        discretionary_power_nodes, power_nodes_edges,
        initial_temperature=initial_temperature, k=k_factor * total_nodes_quantity, weight_attr=weight_attr)

    final_cost = calculate_acc(sa_solution, weight_attr) + calculate_aoc(sa_solution)
    final_average_weight = nx.average_shortest_path_length(sa_solution, weight=weight_attr)

    return sa_solution, initial_cost, final_cost, initial_average_weight, final_average_weight
