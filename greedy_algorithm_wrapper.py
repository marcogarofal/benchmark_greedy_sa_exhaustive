# greedy_algorithm_wrapper.py
"""
Wrapper for greedy algorithm to make it compatible with exhaustive comparison framework
Provides unified interface for algorithm comparison
"""

import networkx as nx
import time
import random


def run_greedy_algorithm(graph_config, algorithm_config, debug_config=None, output_dir='plots_greedy', pre_built_graph=None):
    """
    Run greedy algorithm with exhaustive-compatible interface

    Args:
        graph_config: dict with keys: weak_nodes, power_nodes_mandatory,
                     power_nodes_discretionary, capacities
        algorithm_config: dict with keys: seed, alpha (optional)
        debug_config: dict or DebugConfig (currently unused by greedy)
        output_dir: directory for output (currently unused by greedy)
        pre_built_graph: optional pre-built NetworkX graph (for benchmarks)

    Returns:
        dict with keys: best_tree, execution_time, num_nodes, num_edges,
                       acc_cost, aoc_cost, score, connected_weak,
                       failed_connections, discretionary_used
    """
    # Import greedy module
    import greedy_steiner

    # ⏱️ START TIMING - everything after this is counted
    start_time = time.time()

    # Extract configuration (simple variable assignment - negligible time)
    weak_nodes = graph_config['weak_nodes']
    mandatory_nodes = graph_config['power_nodes_mandatory']  # Different name, same thing
    discretionary_nodes = graph_config['power_nodes_discretionary']  # Different name, same thing
    capacities = graph_config['capacities']

    # Get algorithm parameters
    seed = algorithm_config.get('seed', None)
    alpha = algorithm_config.get('alpha', 0.5)

    # Create or use graph
    if pre_built_graph is not None:
        # Use pre-built graph (for benchmarks)
        graph = pre_built_graph
    else:
        # Create graph (MUST be included in timing - same as exhaustive)
        graph = nx.Graph()

        all_nodes = list(weak_nodes) + list(mandatory_nodes) + list(discretionary_nodes)

        # Add nodes with type attributes (same format as exhaustive)
        for node in weak_nodes:
            graph.add_node(node, node_type='weak')
        for node in mandatory_nodes:
            graph.add_node(node, node_type='power_mandatory')
        for node in discretionary_nodes:
            graph.add_node(node, node_type='power_discretionary')

        # Add edges with weights (complete graph with same seed as exhaustive)
        if seed is not None:
            random.seed(seed)

        for i in all_nodes:
            for j in all_nodes:
                if i < j:  # Avoid duplicate edges
                    weight = random.randint(1, 10)
                    graph.add_edge(i, j, weight=weight)

    # Setup global variables required by greedy algorithm
    greedy_steiner.power_capacities = capacities
    greedy_steiner.main_graph = graph

    # Run greedy algorithm
    best_solution, all_solutions = greedy_steiner.find_best_solution_simplified(
        graph,
        weak_nodes,
        mandatory_nodes,
        discretionary_nodes,
        capacities.copy(),
        alpha
    )

    # ⏱️ STOP TIMING - algorithm execution complete
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Convert solution to standard format (NOT counted in timing)
    result = {
        'best_tree': best_solution.steiner_tree,
        'execution_time': elapsed_time,
        'num_nodes': best_solution.steiner_tree.number_of_nodes() if best_solution.steiner_tree else 0,
        'num_edges': best_solution.steiner_tree.number_of_edges() if best_solution.steiner_tree else 0,

        # Additional metrics specific to greedy (useful for comparison)
        'acc_cost': best_solution.acc_cost,
        'aoc_cost': best_solution.aoc_cost,
        'score': best_solution.score,
        'connected_weak': len(best_solution.connected_weak),
        'failed_connections': len(best_solution.failed_connections),
        'discretionary_used': best_solution.discretionary_used,
        'total_edge_cost': best_solution.total_cost,
        'alpha': alpha,

        # Keep all solutions for detailed analysis if needed
        'all_solutions': all_solutions
    }

    return result


if __name__ == "__main__":
    # Test the wrapper
    print("Testing Greedy Algorithm Wrapper")
    print("=" * 60)

    from config_loader import load_config
    from graph_generator import generate_complete_config

    # Load configuration
    config = load_config('config.json')
    graph_config = generate_complete_config(config)

    # Algorithm configuration
    algorithm_config = {
        'seed': 42,
        'alpha': 0.5  # Balance between ACC and AOC
    }

    print(f"\nGraph Configuration:")
    print(f"  Weak nodes: {graph_config['weak_nodes']}")
    print(f"  Mandatory nodes: {graph_config['power_nodes_mandatory']}")
    print(f"  Discretionary nodes: {graph_config['power_nodes_discretionary']}")
    print(f"  Seed: {algorithm_config['seed']}")
    print(f"  Alpha: {algorithm_config['alpha']}")

    # Run greedy algorithm
    print(f"\nRunning greedy algorithm...")
    result = run_greedy_algorithm(graph_config, algorithm_config)

    # Display results
    print(f"\n{'=' * 60}")
    print("GREEDY ALGORITHM RESULTS")
    print(f"{'=' * 60}")
    print(f"✓ Execution time: {result['execution_time']:.4f}s")
    print(f"  Nodes in solution: {result['num_nodes']}")
    print(f"  Edges in solution: {result['num_edges']}")
    print(f"  Connected weak nodes: {result['connected_weak']}/{len(graph_config['weak_nodes'])}")
    print(f"  Failed connections: {result['failed_connections']}")
    print(f"  Discretionary used: {result['discretionary_used']}")
    print(f"\n  Cost Metrics:")
    print(f"    ACC (communication): {result['acc_cost']:.6f}")
    print(f"    AOC (operational): {result['aoc_cost']:.6f}")
    print(f"    Total score: {result['score']:.2f}")
    print(f"    Edge cost: {result['total_edge_cost']}")
    print(f"\n{'=' * 60}")
