# greedy_steiner.py
"""
Greedy Steiner Tree Algorithm - Core Functions
Extracted for use as a library module
"""

import networkx as nx
import random
from itertools import combinations

# Global variables (required by the algorithm)
power_capacities = {}
main_graph = None


class Solution:
    def __init__(self, steiner_tree, capacity_usage, connected_weak, failed_connections,
                 total_cost, capacity_cost, discretionary_used, graph_info="",
                 acc_cost=0, aoc_cost=0, alpha=0.5):
        self.steiner_tree = steiner_tree
        self.capacity_usage = capacity_usage
        self.connected_weak = connected_weak
        self.failed_connections = failed_connections
        self.total_cost = total_cost
        self.capacity_cost = capacity_cost
        self.discretionary_used = discretionary_used
        self.graph_info = graph_info
        self.acc_cost = acc_cost
        self.aoc_cost = aoc_cost
        self.alpha = alpha
        self.score = self.calculate_score()

    def calculate_cost_function(self, graph, selected_edges, selected_nodes, alpha=0.5):
        """
        MODIFIED: Use EXHAUSTIVE cost function instead of ACC/AOC
        This makes greedy optimize for the same metric as exhaustive
        """
        # Get node types from graph
        weak_nodes = [n for n in graph.nodes() if graph.nodes[n].get('node_type') == 'weak']
        mandatory_nodes = [n for n in graph.nodes() if graph.nodes[n].get('node_type') == 'power_mandatory']
        discretionary_nodes = [n for n in graph.nodes() if graph.nodes[n].get('node_type') == 'power_discretionary']
        
        max_nodes = len(weak_nodes) + len(mandatory_nodes) + len(discretionary_nodes)
        max_possible_edges = max_nodes - 1
        
        # ===== EDGE COST (same as exhaustive) =====
        if not selected_edges:
            return 0, 0, 0
        
        # Calculate global_max_weight from full graph
        global_max_weight = max(graph[u][v]['weight'] for u, v in graph.edges())
        
        edge_weights = [graph[u][v]['weight'] for u, v in selected_edges]
        total_weight = sum(edge_weights)
        max_possible_weight = global_max_weight * max_possible_edges
        
        edge_cost = total_weight / max_possible_weight if max_possible_weight > 0 else 0
        
        # ===== DEGREE COST (same as exhaustive) =====
        all_power = set(mandatory_nodes + discretionary_nodes)
        power_in_tree = [n for n in selected_nodes if n in all_power]
        
        degree_sum = 0
        for node in power_in_tree:
            if node in power_capacities and power_capacities[node] > 0:
                degree = sum(1 for u, v in selected_edges if node in (u, v))
                degree_sum += degree / power_capacities[node]
        
        degree_cost_raw = degree_sum / len(power_in_tree) if power_in_tree else 0
        
        # Normalize to [0,1]
        min_capacity = min(power_capacities[n] for n in all_power if n in power_capacities) if power_capacities else 1
        max_degree_possible = max_nodes - 1
        max_degree_cost_per_node = max_degree_possible / min_capacity if min_capacity > 0 else 1
        
        degree_cost = degree_cost_raw / max_degree_cost_per_node if max_degree_cost_per_node > 0 else 0
        
        # ===== TOTAL COST =====
        total_cost = edge_cost + degree_cost
        
        return total_cost, edge_cost, degree_cost


    def calculate_score(self):
        """Calculate score using exhaustive cost function"""
        selected_nodes = set()
        selected_edges = list(self.steiner_tree.edges())

        for u, v in selected_edges:
            selected_nodes.add(u)
            selected_nodes.add(v)

        try:
            cost_func_value, edge_cost, degree_cost = self.calculate_cost_function(
                main_graph, selected_edges, selected_nodes, self.alpha
            )
            self.acc_cost = edge_cost  # Store for reference
            self.aoc_cost = degree_cost  # Store for reference
            self.weighted_cost = cost_func_value
        except Exception as e:
            print(f"Error calculating cost: {e}")
            cost_func_value = float('inf')
            self.acc_cost = float('inf')
            self.aoc_cost = float('inf')
            self.weighted_cost = float('inf')

        # Penalties for failed connections
        connection_penalty = len(self.failed_connections) * 1000

        connectivity_penalty = 0
        if len(selected_edges) > 0:
            temp_graph = nx.Graph()
            temp_graph.add_edges_from(selected_edges)
            if not nx.is_connected(temp_graph):
                connectivity_penalty = 500

        total_score = cost_func_value * 1000 + connection_penalty + connectivity_penalty

        return total_score



def find_all_paths_to_mandatory(graph, weak_node, mandatory_nodes, discretionary_nodes, max_hops=4):
    """Find ALL possible paths from weak node to mandatory nodes"""
    all_paths = []

    # Direct paths
    for mandatory_node in mandatory_nodes:
        if graph.has_edge(weak_node, mandatory_node):
            cost = graph[weak_node][mandatory_node]['weight']
            all_paths.append({
                'path': [weak_node, mandatory_node],
                'cost': cost,
                'target_mandatory': mandatory_node,
                'discretionary_used': []
            })

    # Paths through 1 discretionary
    for disc_node in discretionary_nodes:
        if graph.has_edge(weak_node, disc_node):
            cost_to_disc = graph[weak_node][disc_node]['weight']

            for mandatory_node in mandatory_nodes:
                if graph.has_edge(disc_node, mandatory_node):
                    total_cost = cost_to_disc + graph[disc_node][mandatory_node]['weight']
                    all_paths.append({
                        'path': [weak_node, disc_node, mandatory_node],
                        'cost': total_cost,
                        'target_mandatory': mandatory_node,
                        'discretionary_used': [disc_node]
                    })

    # Paths through 2 discretionary
    if max_hops >= 3:
        for disc1 in discretionary_nodes:
            if graph.has_edge(weak_node, disc1):
                cost_to_disc1 = graph[weak_node][disc1]['weight']

                for disc2 in discretionary_nodes:
                    if disc1 != disc2 and graph.has_edge(disc1, disc2):
                        cost_disc1_to_disc2 = graph[disc1][disc2]['weight']

                        for mandatory_node in mandatory_nodes:
                            if graph.has_edge(disc2, mandatory_node):
                                total_cost = cost_to_disc1 + cost_disc1_to_disc2 + graph[disc2][mandatory_node]['weight']
                                all_paths.append({
                                    'path': [weak_node, disc1, disc2, mandatory_node],
                                    'cost': total_cost,
                                    'target_mandatory': mandatory_node,
                                    'discretionary_used': [disc1, disc2]
                                })

    all_paths.sort(key=lambda x: x['cost'])
    return all_paths


def solve_with_discretionary_subset(graph, weak_nodes, mandatory_nodes, discretionary_subset,
                                   power_capacities_copy, graph_info="", alpha=0.5):
    """Solve using specific subset of discretionary nodes"""
    steiner_tree = nx.Graph()
    capacity_usage = {node: 0 for node in mandatory_nodes + discretionary_subset}
    connected_weak = set()
    failed_connections = []
    actually_used_discretionary = set()

    # Connect all mandatory nodes first
    if len(mandatory_nodes) > 1:
        mandatory_subgraph = graph.subgraph(mandatory_nodes).copy()

        if nx.is_connected(mandatory_subgraph):
            mandatory_mst = nx.minimum_spanning_tree(mandatory_subgraph, weight='weight')
            for u, v in mandatory_mst.edges():
                steiner_tree.add_edge(u, v, weight=graph[u][v]['weight'])
        else:
            # Connect using shortest paths
            mandatory_set = set(mandatory_nodes)
            connected_mandatory = set([mandatory_nodes[0]])

            while connected_mandatory != mandatory_set:
                best_path = None
                best_cost = float('inf')

                for connected in connected_mandatory:
                    for unconnected in mandatory_set - connected_mandatory:
                        try:
                            path = nx.shortest_path(graph, connected, unconnected, weight='weight')
                            cost = nx.shortest_path_length(graph, connected, unconnected, weight='weight')

                            if cost < best_cost:
                                best_cost = cost
                                best_path = path
                        except nx.NetworkXNoPath:
                            continue

                if best_path:
                    for i in range(len(best_path) - 1):
                        steiner_tree.add_edge(best_path[i], best_path[i+1],
                                            weight=graph[best_path[i]][best_path[i+1]]['weight'])

                        if best_path[i] in discretionary_subset:
                            capacity_usage[best_path[i]] = capacity_usage.get(best_path[i], 0) + 1
                            actually_used_discretionary.add(best_path[i])
                        if best_path[i+1] in discretionary_subset:
                            capacity_usage[best_path[i+1]] = capacity_usage.get(best_path[i+1], 0) + 1
                            actually_used_discretionary.add(best_path[i+1])

                    for node in best_path:
                        if node in mandatory_set:
                            connected_mandatory.add(node)
                else:
                    break

    # Find paths for weak nodes
    all_weak_options = {}
    for weak_node in weak_nodes:
        paths = find_all_paths_to_mandatory(graph, weak_node, mandatory_nodes, discretionary_subset)
        all_weak_options[weak_node] = paths

    # Greedy connection with cost function
    all_options = []
    for weak_node, paths in all_weak_options.items():
        for path_info in paths:
            path_edges = [(path_info['path'][i], path_info['path'][i+1])
                         for i in range(len(path_info['path'])-1)]

            simulated_capacity_usage = capacity_usage.copy()
            simulated_tree_edges = list(steiner_tree.edges())

            target_mandatory = path_info['target_mandatory']
            discretionary_used = path_info['discretionary_used']
            path = path_info['path']

            simulated_capacity_usage[target_mandatory] = simulated_capacity_usage.get(target_mandatory, 0) + 1
            for disc_node in discretionary_used:
                simulated_capacity_usage[disc_node] = simulated_capacity_usage.get(disc_node, 0) + 1

            for i in range(len(path) - 1):
                simulated_tree_edges.append((path[i], path[i+1]))

            simulated_selected_nodes = set()
            for u, v in simulated_tree_edges:
                simulated_selected_nodes.add(u)
                simulated_selected_nodes.add(v)

            edge_weight_sum = sum(graph[u][v]['weight'] for u, v in path_edges)
            n = len(graph.nodes())
            incremental_acc = edge_weight_sum / (n * (n - 1)) if n > 1 else 0

            # Calculate AOC increment (simplified)
            incremental_aoc = 0

            incremental_cost = alpha * incremental_acc + (1 - alpha) * incremental_aoc

            all_options.append({
                'weak_node': weak_node,
                'incremental_cost': incremental_cost,
                'incremental_acc': incremental_acc,
                'incremental_aoc': incremental_aoc,
                'edge_cost': edge_weight_sum,
                **path_info
            })

    all_options.sort(key=lambda x: x['incremental_cost'])

    # Connect weak nodes
    for option in all_options:
        weak_node = option['weak_node']

        if weak_node in connected_weak:
            continue

        path = option['path']
        target_mandatory = option['target_mandatory']
        discretionary_used = option['discretionary_used']

        capacity_usage[target_mandatory] += 1
        for disc_node in discretionary_used:
            capacity_usage[disc_node] += 1
            actually_used_discretionary.add(disc_node)

        for i in range(len(path) - 1):
            steiner_tree.add_edge(path[i], path[i+1], weight=graph[path[i]][path[i+1]]['weight'])

        connected_weak.add(weak_node)

    # Handle remaining weak nodes
    remaining_weak = set(weak_nodes) - connected_weak

    if remaining_weak:
        for weak_node in remaining_weak:
            available_paths = all_weak_options.get(weak_node, [])
            if available_paths:
                chosen_path = available_paths[0]
                target_mandatory = chosen_path['target_mandatory']
                discretionary_used = chosen_path['discretionary_used']
                path = chosen_path['path']

                capacity_usage[target_mandatory] += 1
                for disc_node in discretionary_used:
                    capacity_usage[disc_node] += 1
                    actually_used_discretionary.add(disc_node)

                for i in range(len(path) - 1):
                    steiner_tree.add_edge(path[i], path[i+1], weight=graph[path[i]][path[i+1]]['weight'])

                connected_weak.add(weak_node)
            else:
                failed_connections.append(weak_node)

    # Calculate statistics
    total_cost = sum(graph[u][v]['weight'] for u, v in steiner_tree.edges())

    capacity_cost = 0
    nodes_actually_used = [n for n in capacity_usage if capacity_usage[n] > 0 and power_capacities_copy.get(n, 0) > 0]

    if nodes_actually_used:
        capacity_ratios = []
        for node in nodes_actually_used:
            if power_capacities_copy[node] > 0:
                ratio = capacity_usage[node] / power_capacities_copy[node]
                capacity_ratios.append(ratio)

        capacity_cost = sum(capacity_ratios) / len(capacity_ratios)

    actually_used_list = sorted(list(actually_used_discretionary))

    return Solution(steiner_tree, capacity_usage, connected_weak, failed_connections,
                   total_cost, capacity_cost, actually_used_list, graph_info, alpha=alpha)


def find_best_solution_simplified(graph, weak_nodes, mandatory_nodes, all_discretionary_nodes,
                                 power_capacities_copy, alpha=0.5):
    """Find best solution by testing without and with discretionary nodes"""
    global main_graph, power_capacities
    main_graph = graph
    power_capacities = power_capacities_copy

    all_solutions = []

    # Solution without discretionary
    solution_no_disc = solve_with_discretionary_subset(
        graph, weak_nodes, mandatory_nodes, [], power_capacities_copy.copy(),
        "WITHOUT discretionary", alpha
    )
    all_solutions.append(solution_no_disc)

    # Solution with all discretionary
    solution_all_disc = solve_with_discretionary_subset(
        graph, weak_nodes, mandatory_nodes, all_discretionary_nodes, power_capacities_copy.copy(),
        f"WITH ALL discretionary", alpha
    )
    all_solutions.append(solution_all_disc)

    # Sort by score
    all_solutions.sort(key=lambda s: s.score)
    best_solution = all_solutions[0]

    print(f"Solution WITHOUT disc: ACC={solution_no_disc.acc_cost:.4f}, AOC={solution_no_disc.aoc_cost:.4f}, Score={solution_no_disc.score:.2f}")
    print(f"Solution WITH disc: ACC={solution_all_disc.acc_cost:.4f}, AOC={solution_all_disc.aoc_cost:.4f}, Score={solution_all_disc.score:.2f}")
    print(f"Winner: {'NO DISC' if best_solution == solution_no_disc else 'WITH DISC'}")

    return best_solution, all_solutions
