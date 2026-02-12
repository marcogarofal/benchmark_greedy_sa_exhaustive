# tree_optimizer_memory_optimized.py
"""
Tree Optimization Algorithm - Memory Optimized Version
Uses generators to avoid loading all combinations in memory
Suitable for graphs with 10+ nodes
SILENT MODE: No prints unless verbose=True in debug config

FIXED:
  1. Degree cost normalized by power nodes PRESENT in the tree
  2. Edge cost uses total weight normalization (penalizes larger trees)
  3. Both costs normalized to [0,1] range for balanced comparison
"""

import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, chain
import random
from collections import Counter
import copy
import time
import os


class DebugConfig:
    """Debug configuration to replace global debug variables"""
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = {}
        self.plot_initial_graphs = config_dict.get('plot_initial_graphs', False)
        self.plot_intermediate = config_dict.get('plot_intermediate', False)
        self.plot_final = config_dict.get('plot_final', False)
        self.save_plots = config_dict.get('save_plots', False)
        self.verbose = config_dict.get('verbose', False)
        self.verbose_level2 = config_dict.get('verbose_level2', False)
        self.verbose_level3 = config_dict.get('verbose_level3', False)


class CombinationGraph:
    def __init__(self, graph, weak_nodes, debug_config):
        if not nx.is_connected(graph):
            raise ValueError("\tThe graph is not fully connected.")
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.weak_nodes = weak_nodes
        self.debug_config = debug_config
        self.all_possible_links = list(self._generate_combinations_mod(graph.nodes, 2))
        self._num_nodes_for_combinations = self.num_nodes - 1

    def _generate_combinations_mod(self, elements, r):
        """Generator for combinations, filtering weak-weak connections"""
        for combo in combinations(elements, r):
            if combo[0] in self.weak_nodes and combo[1] in self.weak_nodes:
                pass
            else:
                yield combo

    def remove_trees_where_weak_nodes_are_hubs(self, weak_nodes, list_tree):
        valori = list(chain(*list_tree))
        count = Counter(valori)
        for weak_node in weak_nodes:
            if count[weak_node] > 1:
                return False
        return True

    def are_discretionary_nodes_singularly_connected(self, edges, discretionary_nodes, check_only_discretionary, check_no_mandatory):
        if check_only_discretionary:
            return True
        graph = {}
        for edge in edges:
            u, v = edge
            if u not in graph:
                graph[u] = []
            if v not in graph:
                graph[v] = []
            graph[u].append(v)
            graph[v].append(u)
        return all(len(graph[node]) > 1 for node in discretionary_nodes if node in graph)


    def filter_combinations_discretionary(self, weak_nodes, power_nodes_mandatory, power_nodes_discretionary, check_only_discretionary, check_no_mandatory):
        """
        Generator che yielda TUTTE le combinazioni valide di spanning trees
        
        FIXED: Genera TUTTE le combinazioni di n-1 archi che formano spanning tree validi
        ✅ Memory optimized: processa un tree alla volta  
        ✅ EXHAUSTIVE: Testa TUTTE le combinazioni possibili
        """
        list_nodes_graph = list(self.graph.nodes())
        num_nodes = len(list_nodes_graph)
        
        if num_nodes == 0:
            return
        
        num_edges_needed = num_nodes - 1  # Spanning tree ha n-1 archi
        
        # Genera TUTTE le combinazioni di n-1 archi
        combinations_gen = combinations(self.all_possible_links, num_edges_needed)
        
        tested_count = 0
        valid_count = 0
        
        for combination in combinations_gen:
            tested_count += 1
            
            if not combination:
                continue
            
            # Verifica 1: Weak nodes non devono essere hub
            if power_nodes_discretionary is not None:
                if not self.remove_trees_where_weak_nodes_are_hubs(weak_nodes, combination):
                    continue
                
                # Verifica 2: Discretionary nodes devono essere connessi correttamente
                if not self.are_discretionary_nodes_singularly_connected(
                    combination, power_nodes_discretionary,
                    check_only_discretionary, check_no_mandatory
                ):
                    continue
            else:
                if not self.remove_trees_where_weak_nodes_are_hubs(weak_nodes, combination):
                    continue
            
            # Verifica 3: Deve formare un grafo connesso (spanning tree)
            if not is_connected(combination, num_nodes):
                continue
            
            valid_count += 1
            
            if self.debug_config.verbose:
                if valid_count <= 5 or valid_count % 10 == 0:
                    print(f"    Valid tree #{valid_count}: {combination}")
            
            yield combination
        
        # Debug finale
        if self.debug_config.verbose:
            print(f"  Spanning trees: tested {tested_count}, valid {valid_count}")




def is_connected(graph, number_of_nodes):
    """Verify if the graph is connected using DFS"""
    def dfs_modified(graph, start):
        visited = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                neighbours = [neighbour for edge in graph for neighbour in edge if node in edge and neighbour != node]
                stack.extend(neighbour for neighbour in neighbours if neighbour not in visited)
        return visited

    start_node = graph[0][0]
    reachable_nodes = dfs_modified(graph, start_node)
    return len(reachable_nodes) == number_of_nodes


def create_graph(weak_nodes=None, power_nodes_mandatory=None, power_nodes_discretionary=None, seed=None):
    """Create a complete graph with random weights"""
    if seed is not None:
        random.seed(seed)

    G = nx.Graph()
    all_nodes = []

    if (weak_nodes is not None) and (power_nodes_mandatory is not None) and (power_nodes_discretionary is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_mandatory, node_type='power_mandatory')
        G.add_nodes_from(power_nodes_discretionary, node_type='power_discretionary')
        all_nodes = list(weak_nodes) + list(power_nodes_mandatory) + list(power_nodes_discretionary)
    elif (weak_nodes is not None) and (power_nodes_mandatory is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_mandatory, node_type='power_mandatory')
        all_nodes = list(weak_nodes) + list(power_nodes_mandatory)
    elif (weak_nodes is not None) and (power_nodes_discretionary is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_discretionary, node_type='power_discretionary')
        all_nodes = list(weak_nodes) + list(power_nodes_discretionary)
    elif power_nodes_discretionary is not None:
        G.add_nodes_from(power_nodes_discretionary, node_type='power_discretionary')
        all_nodes = list(power_nodes_discretionary)

    for i in all_nodes:
        for j in all_nodes:
            if i != j and not G.has_edge(i, j):
                weight = random.randint(1, 10)
                G.add_edge(i, j, weight=weight)
    return G


def draw_graph(G):
    """Draw graph with colored nodes"""
    plt.clf()
    pos = nx.spring_layout(G)
    node_colors = {'weak': 'green', 'power_mandatory': 'red', 'power_discretionary': 'orange'}
    colors = [node_colors[data['node_type']] for _, data in G.nodes(data=True)]
    edge_labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}

    nx.draw(G, pos, with_labels=True, node_color=colors, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()


def save_graph(G, path, count_picture, name=None, edge_cost=None, degree_cost=None):
    """Save graph to file with optional score annotation"""
    pos = nx.spring_layout(G)
    node_colors = {'weak': 'green', 'power_mandatory': 'red', 'power_discretionary': 'orange'}
    colors = [node_colors[data['node_type']] for _, data in G.nodes(data=True)]
    edge_labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}

    fig, ax = plt.subplots(figsize=(10, 8))

    nx.draw(G, pos, with_labels=True, node_color=colors, font_weight='bold', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    if edge_cost is not None and degree_cost is not None:
        total_cost = edge_cost + degree_cost
        score_text = f'Edge Cost: {edge_cost:.4f}\nDegree Cost: {degree_cost:.4f}\nTotal: {total_cost:.4f}'
        ax.text(0.02, 0.98, score_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        filename = f"{count_picture}_{name}_e{edge_cost:.4f}_d{degree_cost:.4f}_t{total_cost:.4f}.png"
    else:
        if name is None:
            filename = f"{count_picture}_graph.png"
        else:
            filename = f"{count_picture}_{name}.png"

    path_to_save = os.path.join(path, filename)
    plt.savefig(path_to_save)
    plt.close()
    return count_picture + 1, filename


def build_tree_from_list_edges(G, desired_edges, no_plot=None):
    """Build tree from list of edges"""
    G_copy = copy.deepcopy(G)
    edges_to_remove = [edge for edge in G_copy.edges() if edge not in desired_edges]
    G_copy.remove_edges_from(edges_to_remove)

    if no_plot is False or no_plot is None:
        pos = nx.spring_layout(G_copy)
        node_colors = {'weak': 'green', 'power_mandatory': 'red', 'power_discretionary': 'orange'}
        colors = [node_colors[data['node_type']] for _, data in G_copy.nodes(data=True)]
        edge_labels = {(i, j): G_copy[i][j].get('weight', None) for i, j in G_copy.edges()}

        nx.draw(G_copy, pos, with_labels=True, node_color=colors, font_weight='bold')
        nx.draw_networkx_edge_labels(G_copy, pos, edge_labels=edge_labels)
        nx.draw_networkx_edges(G_copy, pos, edgelist=desired_edges)
        plt.show()
    return G_copy


def get_weight(item):
    """Helper function for max weight extraction"""
    return item[1]['weight']


def save_scores_to_json(scores, output_path):
    """Save scores dictionary to JSON file"""
    import json
    with open(output_path, 'w') as f:
        json.dump(scores, f, indent=2)


def calculate_absolute_scores(trees_data, capacities, power_nodes_mandatory, power_nodes_discretionary, weak_nodes):
    """Calculate absolute scores for all trees with global normalization"""
    if not trees_data:
        return {}

    global_max_weight = 0

    for tree, _ in trees_data:
        if tree and tree.number_of_edges() > 0:
            edges_with_weights = [(edge, tree.get_edge_data(edge[0], edge[1])) for edge in tree.edges()]
            tree_max_weight = max(edges_with_weights, key=get_weight)[1]["weight"]
            global_max_weight = max(global_max_weight, tree_max_weight)

    if global_max_weight == 0:
        global_max_weight = 1

    # Calcola costanti per normalizzazione
    max_nodes = len(weak_nodes) + len(power_nodes_mandatory) + len(power_nodes_discretionary)
    max_possible_edges = max_nodes - 1
    max_possible_weight = global_max_weight * max_possible_edges
    
    all_power = set(list(power_nodes_mandatory) + list(power_nodes_discretionary))
    min_capacity = min(capacities[n] for n in all_power if n in capacities) if capacities and all_power else 1
    max_degree_possible = max_nodes - 1
    max_degree_cost_per_node = max_degree_possible / min_capacity if min_capacity > 0 else 1

    scores = {
        "global_max_weight": global_max_weight,
        "max_possible_weight": max_possible_weight,
        "max_degree_cost_per_node": max_degree_cost_per_node,
        "trees": {}
    }

    for tree, filename in trees_data:
        if tree and tree.number_of_edges() > 0:
            edges_with_weights = [(edge, tree.get_edge_data(edge[0], edge[1])) for edge in tree.edges()]
            total_weight = sum(data['weight'] for edge, data in edges_with_weights)

            # Edge cost normalizzato
            edge_cost = total_weight / max_possible_weight if max_possible_weight > 0 else 0

            # Degree cost normalizzato
            power_in_tree = [n for n in tree.nodes() if n in all_power]
            degree_sum = sum(tree.degree(n) / capacities[n] for n in power_in_tree if n in capacities)
            degree_cost_raw = degree_sum / len(power_in_tree) if power_in_tree else 0
            degree_cost = degree_cost_raw / max_degree_cost_per_node if max_degree_cost_per_node > 0 else 0

            total_cost = edge_cost + degree_cost

            scores["trees"][filename] = {
                "edge_cost": round(edge_cost, 4),
                "degree_cost": round(degree_cost, 4),
                "total": round(total_cost, 4),
                "num_nodes": tree.number_of_nodes(),
                "num_edges": tree.number_of_edges(),
                "nodes": list(tree.nodes()),
                "edges": [list(edge) for edge in tree.edges()]
            }
        else:
            scores["trees"][filename] = {
                "edge_cost": 0,
                "degree_cost": 0,
                "total": 0,
                "num_nodes": 0,
                "num_edges": 0,
                "nodes": [],
                "edges": []
            }

    return scores


def compare_2_trees(tree1, tree2, power_nodes_mandatory, power_nodes_discretionary, weak_nodes, capacities, debug_config):
    """
    Compare two trees and return the best one with its costs
    
    FIXED:
      1. Degree cost normalized by power nodes PRESENT in the tree
      2. Edge cost uses total weight normalization (penalizes larger trees)
      3. Both costs normalized to [0,1] for balanced comparison
    """
    if tree1 is None:
        tree1 = tree2

    all_power = set(list(power_nodes_mandatory) + list(power_nodes_discretionary))
    
    # Identifica power nodes effettivi in ogni albero
    power_in_tree1 = [n for n in tree1.nodes() if n in all_power]
    power_in_tree2 = [n for n in tree2.nodes() if n in all_power]

    # ===== EDGE COST: Peso totale normalizzato =====
    edges_with_weights1 = [(edge, tree1.get_edge_data(edge[0], edge[1])) for edge in tree1.edges()]
    edges_with_weights2 = [(edge, tree2.get_edge_data(edge[0], edge[1])) for edge in tree2.edges()]
    
    max_edge_cost1 = max(edges_with_weights1, key=get_weight)[1]["weight"] if edges_with_weights1 else 1
    max_edge_cost2 = max(edges_with_weights2, key=get_weight)[1]["weight"] if edges_with_weights2 else 1
    global_max_weight = max(max_edge_cost1, max_edge_cost2)

    total_weight1 = sum(data['weight'] for edge, data in edges_with_weights1)
    total_weight2 = sum(data['weight'] for edge, data in edges_with_weights2)

    # Massimo peso teorico: albero con tutti i nodi
    max_nodes = len(weak_nodes) + len(power_nodes_mandatory) + len(power_nodes_discretionary)
    max_possible_edges = max_nodes - 1
    max_possible_weight = global_max_weight * max_possible_edges

    edgecost1 = total_weight1 / max_possible_weight if max_possible_weight > 0 else 0
    edgecost2 = total_weight2 / max_possible_weight if max_possible_weight > 0 else 0

    # ===== DEGREE COST: Utilizzo medio delle capacità =====
    degree_sum1 = 0
    for x in power_in_tree1:
        if x in capacities:
            try:
                degree_sum1 += tree1.degree(x) / capacities[x]
            except (AttributeError, KeyError):
                if debug_config.verbose:
                    print("error in degree_sum1")

    degree_sum2 = 0
    for x in power_in_tree2:
        if x in capacities:
            try:
                degree_sum2 += tree2.degree(x) / capacities[x]
            except (AttributeError, KeyError):
                if debug_config.verbose:
                    print("error in degree_sum2")

    # Normalizza per i power nodes presenti nell'albero
    cost_degree1_raw = degree_sum1 / len(power_in_tree1) if power_in_tree1 else 0
    cost_degree2_raw = degree_sum2 / len(power_in_tree2) if power_in_tree2 else 0

    # Normalizza a [0,1] dividendo per il caso peggiore teorico
    # Caso peggiore: stella con nodo di capacità minima al centro
    min_capacity = min(capacities[n] for n in all_power if n in capacities) if capacities and all_power else 1
    max_degree_possible = max_nodes - 1
    max_degree_cost_per_node = max_degree_possible / min_capacity if min_capacity > 0 else 1

    cost_degree1 = cost_degree1_raw / max_degree_cost_per_node if max_degree_cost_per_node > 0 else 0
    cost_degree2 = cost_degree2_raw / max_degree_cost_per_node if max_degree_cost_per_node > 0 else 0

    # ===== CONFRONTO =====
    total_cost1 = edgecost1 + cost_degree1
    total_cost2 = edgecost2 + cost_degree2

    if total_cost1 <= total_cost2:
        return tree1, edgecost1, cost_degree1, True
    else:
        return tree2, edgecost2, cost_degree2, False


def join_2_trees(graph1, graph2, weak_nodes, power_nodes_mandatory, power_nodes_discretionary, added_edges, seed=None, pre_built_graph=None):
    """Join two trees into one graph"""
    if seed is not None:
        random.seed(seed)

    G = nx.Graph()
    all_nodes = []

    if (weak_nodes is not None) and (power_nodes_mandatory is not None) and (power_nodes_discretionary is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_mandatory, node_type='power_mandatory')
        G.add_nodes_from(power_nodes_discretionary, node_type='power_discretionary')
        all_nodes = list(weak_nodes) + list(power_nodes_mandatory) + list(power_nodes_discretionary)
    elif (weak_nodes is not None) and (power_nodes_mandatory is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_mandatory, node_type='power_mandatory')
        all_nodes = list(weak_nodes) + list(power_nodes_mandatory)
    elif power_nodes_discretionary is not None:
        G.add_nodes_from(power_nodes_discretionary, node_type='power_discretionary')
        all_nodes = list(power_nodes_discretionary)

    for i in all_nodes:
        for j in all_nodes:
            if i != j and not G.has_edge(i, j):
                matching_tuple = next((tup for tup in added_edges if set(tup[:2]) == set((i, j))), None)

                if pre_built_graph is not None and pre_built_graph.has_edge(i, j):
                    G.add_edge(i, j, weight=pre_built_graph[i][j]['weight'])
                elif (graph1.has_edge(i, j) or graph1.has_edge(j, i)):
                    G.add_edge(i, j, weight=graph1[i][j]['weight'])
                elif (graph2.has_edge(i, j) or graph2.has_edge(j, i)):
                    G.add_edge(i, j, weight=graph2[i][j]['weight'])
                elif matching_tuple:
                    weight_value = next(iter(matching_tuple[2]))
                    G.add_edge(i, j, weight=weight_value)
                else:
                    weight = random.randint(1, 10)
                    G.add_edge(i, j, weight=weight)
                    new_element = (i, j, frozenset({weight}))
                    added_edges.add(new_element)
    return G



def generate_graphs(graph, power_nodes_discretionary, weak_nodes, power_nodes_mandatory, added_edges, debug_config, seed=None, pre_built_graph=None):
    """
    Generator that yields graphs with different discretionary node combinations
    
    FIXED: Now correctly generates ALL combinations including:
    - Empty set (no discretionary nodes)
    - All individual nodes
    - All pairs
    - All triplets
    - ... up to all discretionary nodes
    
    For n discretionary nodes, generates 2^n combinations.
    """
    if pre_built_graph is not None:
        graph2 = pre_built_graph.subgraph(power_nodes_discretionary).copy()
    else:
        graph2 = create_graph(power_nodes_discretionary=power_nodes_discretionary, seed=seed)

    if debug_config.plot_initial_graphs:
        draw_graph(graph2)

    def generate_combinations(elements):
        """
        Generate ALL combinations from 0 to n nodes
        
        FIXED: Changed from range(1, n+1) to range(0, n+1)
        This ensures we test the case with NO discretionary nodes
        """
        number_of_nodes = len(elements)
        # CRITICAL FIX: Start from 0 to include empty set
        for x in range(0, number_of_nodes + 1):  # Was: range(1, number_of_nodes + 1)
            for combo in combinations(elements, x):
                yield combo

    # Calculate expected number of combinations for validation
    num_discretionary = len(power_nodes_discretionary)
    expected_combos = 2 ** num_discretionary  # 2^n combinations
    
    if debug_config.verbose:
        print(f"\n{'='*80}")
        print(f"DISCRETIONARY COMBINATIONS FOR GRAPH GENERATION")
        print(f"{'='*80}")
        print(f"Discretionary nodes available: {power_nodes_discretionary}")
        print(f"Number of discretionary nodes: {num_discretionary}")
        print(f"Expected total combinations: {expected_combos} (2^{num_discretionary})")
        print(f"{'='*80}\n")
    
    combinations_only_power_nodes_discretionary = generate_combinations(power_nodes_discretionary)
    count = 0

    for combo in combinations_only_power_nodes_discretionary:
        # Convert tuple to list of nodes
        lista_risultante = list(combo)
        
        if debug_config.verbose:
            if len(lista_risultante) == 0:
                print(f"Combination {count + 1}/{expected_combos}: [] (no discretionary)")
            else:
                print(f"Combination {count + 1}/{expected_combos}: {lista_risultante}")

        # Build graph with this combination of discretionary nodes
        if count == 0:
            # First iteration: create initial graph
            graph3 = join_2_trees(
                graph, graph2, 
                weak_nodes=weak_nodes,
                power_nodes_mandatory=power_nodes_mandatory,
                power_nodes_discretionary=lista_risultante,
                added_edges=added_edges, 
                seed=seed, 
                pre_built_graph=pre_built_graph
            )
            graph3_bak = graph3  # Keep backup for subsequent iterations
        else:
            # Subsequent iterations: rebuild from backup
            graph3 = join_2_trees(
                graph3_bak, graph2, 
                weak_nodes=weak_nodes,
                power_nodes_mandatory=power_nodes_mandatory,
                power_nodes_discretionary=lista_risultante,
                added_edges=added_edges, 
                seed=seed, 
                pre_built_graph=pre_built_graph
            )

        count += 1
        
        if debug_config.verbose:
            total_nodes = len(weak_nodes) + len(power_nodes_mandatory) + len(lista_risultante)
            print(f"  → Graph: {graph3.number_of_nodes()} nodes ({total_nodes} expected), "
                  f"{graph3.number_of_edges()} edges\n")
        
        yield graph3
    
    # Final validation
    if debug_config.verbose:
        print(f"{'='*80}")
        print(f"GENERATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total graphs generated: {count}")
        print(f"Expected: {expected_combos}")
        if count == expected_combos:
            print(f"✅ SUCCESS: All {expected_combos} combinations tested!")
        else:
            print(f"❌ ERROR: Missing {expected_combos - count} combinations!")
        print(f"{'='*80}\n")


def process_graph(graph, weak_nodes, power_nodes_mandatory, power_nodes_discretionary,
                 best_tree, best_edge_cost, best_degree_cost, capacities,
                 check_only_discretionary, check_no_mandatory,
                 debug_config, plot_path, count_picture, trees_registry):
    """Process a graph to find optimal tree - Memory optimized with generator"""
    if debug_config.plot_intermediate:
        draw_graph(graph)

    combinations_graph = CombinationGraph(graph, weak_nodes, debug_config)
    if debug_config.verbose:
        input("\n\nENTER to continue...")

    valid_combinations_gen = combinations_graph.filter_combinations_discretionary(
        weak_nodes, power_nodes_mandatory,
        power_nodes_discretionary,
        check_only_discretionary, check_no_mandatory
    )
    
    for x in valid_combinations_gen:
        if debug_config.verbose:
            print("nodes: ", x)

        if debug_config.plot_intermediate:
            tree = build_tree_from_list_edges(graph, x, no_plot=False)
        else:
            tree = build_tree_from_list_edges(graph, x, no_plot=True)

        best_tree, best_edge_cost, best_degree_cost, is_best = compare_2_trees(
            best_tree, tree, power_nodes_mandatory,
            power_nodes_discretionary, weak_nodes, capacities, debug_config)

        if debug_config.save_plots:
            count_picture, filename = save_graph(tree, plot_path, count_picture, "intermediate")
            trees_registry.append((copy.deepcopy(tree), filename))

        if debug_config.verbose:
            input("\n\nENTER to continue...")
        if debug_config.plot_intermediate:
            draw_graph(best_tree)


    if debug_config.verbose:
        print(f"\n=== END OF COMBINATION ===")
        print(f"Best tree for this combination:")
        print(f"  Nodes: {best_tree.number_of_nodes()}")
        print(f"  Edge cost: {best_edge_cost:.6f}")
        print(f"  Degree cost: {best_degree_cost:.6f}")
        print(f"  TOTAL: {best_edge_cost + best_degree_cost:.6f}")

    return best_tree, best_edge_cost, best_degree_cost, count_picture


def run_algorithm(graph_config, algorithm_config, debug_config=None, output_dir='plots', pre_built_graph=None):
    """
    Main entry point for the tree optimization algorithm - Memory optimized version
    SILENT MODE: No output unless verbose=True
    """
    start_time = time.time()

    if debug_config is None:
        debug_config = DebugConfig()
    elif isinstance(debug_config, dict):
        debug_config = DebugConfig(debug_config)

    weak_nodes = graph_config['weak_nodes']
    power_nodes_mandatory = graph_config['power_nodes_mandatory']
    power_nodes_discretionary = graph_config['power_nodes_discretionary']
    capacities = graph_config['capacities']
    seed = algorithm_config.get('seed', None)

    plot_path = output_dir
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    else:
        for filename in os.listdir(plot_path):
            file_path = os.path.join(plot_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                if debug_config.verbose:
                    print(f"Unable to delete {file_path}: {e}")

    count_picture = 0
    added_edges = set()
    trees_registry = []

    num_nodes = len(weak_nodes) + len(power_nodes_mandatory) + len(power_nodes_discretionary)
    check_only_discretionary = len(power_nodes_discretionary) == num_nodes
    check_no_mandatory = len(power_nodes_mandatory) == 0 or len(power_nodes_mandatory) == 1

    # Build initial graph (weak + mandatory only)
    if len(weak_nodes) + len(power_nodes_mandatory) > 0:
        if pre_built_graph is not None:
            graph = pre_built_graph.subgraph(list(weak_nodes) + list(power_nodes_mandatory)).copy()
        else:
            graph = create_graph(weak_nodes, power_nodes_mandatory, power_nodes_discretionary=None, seed=seed)

        if debug_config.plot_initial_graphs:
            draw_graph(graph)
        if debug_config.save_plots:
            count_picture, filename = save_graph(graph, plot_path, count_picture)

        combinations_graph = CombinationGraph(graph, weak_nodes, debug_config)
        
        valid_combinations_gen = combinations_graph.filter_combinations_discretionary(
            weak_nodes, power_nodes_mandatory, None,
            check_only_discretionary, check_no_mandatory
        )
        
        valid_combinations_list = list(valid_combinations_gen)

        if len(valid_combinations_list) > 0:
            # Inizializza best_tree con None e trova il migliore iterando
            best_tree = None
            best_edge_cost = 0
            best_degree_cost = 0
            
            for x in valid_combinations_list:
                if debug_config.verbose:
                    print("nodes: ", x)

                if debug_config.plot_intermediate:
                    tree = build_tree_from_list_edges(graph, x, no_plot=False)
                else:
                    tree = build_tree_from_list_edges(graph, x, no_plot=True)

                if best_tree is None:
                    best_tree = tree
                    # Calcola costi iniziali
                    edges_with_weights = [(edge, best_tree.get_edge_data(edge[0], edge[1])) for edge in best_tree.edges()]
                    if edges_with_weights:
                        max_edge_cost = max(edges_with_weights, key=get_weight)[1]["weight"]
                        total_weight = sum(data['weight'] for edge, data in edges_with_weights)
                        
                        max_nodes = len(weak_nodes) + len(power_nodes_mandatory) + len(power_nodes_discretionary)
                        max_possible_edges = max_nodes - 1
                        max_possible_weight = max_edge_cost * max_possible_edges
                        best_edge_cost = total_weight / max_possible_weight if max_possible_weight > 0 else 0
                        
                        all_power = set(list(power_nodes_mandatory) + list(power_nodes_discretionary))
                        power_in_tree = [n for n in best_tree.nodes() if n in all_power]
                        degree_sum = sum(best_tree.degree(n) / capacities[n] for n in power_in_tree if n in capacities)
                        degree_cost_raw = degree_sum / len(power_in_tree) if power_in_tree else 0
                        
                        min_capacity = min(capacities[n] for n in all_power if n in capacities) if capacities and all_power else 1
                        max_degree_possible = max_nodes - 1
                        max_degree_cost_per_node = max_degree_possible / min_capacity if min_capacity > 0 else 1
                        best_degree_cost = degree_cost_raw / max_degree_cost_per_node if max_degree_cost_per_node > 0 else 0
                    else:
                        best_edge_cost = 0
                        best_degree_cost = 0
                else:
                    best_tree, best_edge_cost, best_degree_cost, is_best = compare_2_trees(
                        best_tree, tree, power_nodes_mandatory,
                        power_nodes_discretionary, weak_nodes, capacities, debug_config)

                if debug_config.save_plots:
                    count_picture, filename = save_graph(tree, plot_path, count_picture, "first_phase")
                    trees_registry.append((copy.deepcopy(tree), filename))

                if debug_config.verbose:
                    input("Enter...")
                if debug_config.plot_intermediate:
                    draw_graph(best_tree)
        else:
            best_tree = None
            best_edge_cost = 0
            best_degree_cost = 0
    else:
        graph = nx.Graph()
        best_tree = None
        best_edge_cost = 0
        best_degree_cost = 0

    # Process discretionary nodes
    graphs = generate_graphs(graph, power_nodes_discretionary, weak_nodes, power_nodes_mandatory,
                            added_edges, debug_config, seed=seed, pre_built_graph=pre_built_graph)

    for graph_iter in graphs:
        best_tree, best_edge_cost, best_degree_cost, count_picture = process_graph(
            graph_iter, weak_nodes, power_nodes_mandatory,
            power_nodes_discretionary, best_tree, best_edge_cost, best_degree_cost,
            capacities, check_only_discretionary, check_no_mandatory,
            debug_config, plot_path, count_picture, trees_registry)

    end_time = time.time()
    elapsed_time = end_time - start_time

    if debug_config.plot_final:
        draw_graph(best_tree)

    best_filename = None
    if debug_config.save_plots and best_tree is not None:
        count_picture, best_filename = save_graph(best_tree, plot_path, count_picture, name="best_tree",
                  edge_cost=best_edge_cost, degree_cost=best_degree_cost)
        trees_registry.append((copy.deepcopy(best_tree), best_filename))

    if debug_config.save_plots and trees_registry:
        scores = calculate_absolute_scores(trees_registry, capacities,
                                          power_nodes_mandatory, power_nodes_discretionary, weak_nodes)
        if best_tree is not None and best_filename and best_filename in scores["trees"]:
            scores["trees"][best_filename]["is_best"] = True

        scores_path = os.path.join(plot_path, "scores.json")
        save_scores_to_json(scores, scores_path)

    return {
        'best_tree': best_tree,
        'execution_time': elapsed_time,
        'num_nodes': best_tree.number_of_nodes() if best_tree else 0,
        'num_edges': best_tree.number_of_edges() if best_tree else 0
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--config':
        from config_loader import load_config
        from graph_generator import generate_complete_config

        config = load_config(sys.argv[2] if len(sys.argv) > 2 else 'config.json')
        graph_config = generate_complete_config(config)
        algorithm_config = config.get('algorithm', {})
        debug_dict = config.get('debug', {})
        output_dir = config.get('output', {}).get('plots_dir', 'plots')

        result = run_algorithm(graph_config, algorithm_config, debug_dict, output_dir)
    else:
        # Default test
        num_nodes = 8
        num_weak_nodes = int(0.4 * num_nodes)
        num_power_nodes_mandatory = int(0.2 * num_nodes)
        num_power_nodes_discretionary = num_nodes - num_weak_nodes - num_power_nodes_mandatory

        weak_nodes = list(range(1, num_weak_nodes + 1))
        power_nodes_mandatory = list(range(num_weak_nodes + 1, num_weak_nodes + num_power_nodes_mandatory + 1))
        power_nodes_discretionary = list(range(num_weak_nodes + num_power_nodes_mandatory + 1,
                                              num_weak_nodes + num_power_nodes_mandatory + num_power_nodes_discretionary + 1))

        capacities = {i: 10 for i in range(1, 21)}
        capacities.update({2: 30, 3: 2, 4: 1, 6: 4, 7: 5, 8: 5})

        graph_config = {
            'weak_nodes': weak_nodes,
            'power_nodes_mandatory': power_nodes_mandatory,
            'power_nodes_discretionary': power_nodes_discretionary,
            'capacities': capacities
        }
        algorithm_config = {'seed': 42}
        debug_dict = {'plot_final': True}

        result = run_algorithm(graph_config, algorithm_config, debug_dict)
        print(f"\nResult: {result}")