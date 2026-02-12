# sa_algorithm_wrapper.py
"""
Wrapper for Simulated Annealing algorithm to make it compatible with comparison framework
Provides unified interface for algorithm comparison
"""

import networkx as nx
import time
import random


def run_sa_algorithm(graph_config, algorithm_config, debug_config=None, output_dir='plots_sa', pre_built_graph=None):
    """
    Run Simulated Annealing algorithm with exhaustive-compatible interface

    Args:
        graph_config: dict with keys: weak_nodes, power_nodes_mandatory,
                     power_nodes_discretionary, capacities
        algorithm_config: dict with keys: seed, initial_temperature (optional), k_factor (optional)
        debug_config: dict or DebugConfig (currently unused by SA)
        output_dir: directory for output (currently unused by SA)
        pre_built_graph: optional pre-built NetworkX graph (for benchmarks)

    Returns:
        dict with keys: best_tree, execution_time, num_nodes, num_edges,
                       initial_cost, final_cost, initial_avg_weight, final_avg_weight
    """
    # Import SA module
    import simulated_annealing_steiner as sa

    # ⏱️ START TIMING
    start_time = time.time()

    # Extract configuration
    constrained_nodes = graph_config['weak_nodes']  # SA calls them "constrained"
    mandatory_power_nodes = graph_config['power_nodes_mandatory']
    discretionary_power_nodes = graph_config['power_nodes_discretionary']
    capacities = graph_config['capacities']

    # Get algorithm parameters
    seed = algorithm_config.get('seed', None)
    initial_temperature = algorithm_config.get('initial_temperature', 120)
    k_factor = algorithm_config.get('k_factor', 12)

    # Create or use graph
    if pre_built_graph is not None:
        # Use pre-built graph (for benchmarks)
        network_graph = pre_built_graph
    else:
        # Create graph (SAME as exhaustive - included in timing)
        if seed is not None:
            random.seed(seed)

        network_graph = nx.Graph()
        all_nodes = list(constrained_nodes) + list(mandatory_power_nodes) + list(discretionary_power_nodes)

        # Add nodes with type and capacity attributes
        for node in constrained_nodes:
            network_graph.add_node(node, node_type='constrained', links=0, capacity=capacities.get(node, 1))
        for node in mandatory_power_nodes:
            network_graph.add_node(node, node_type='power_mandatory', links=0, capacity=capacities.get(node, 10))
        for node in discretionary_power_nodes:
            network_graph.add_node(node, node_type='power_discretionary', links=0, capacity=capacities.get(node, 10))

        # Add edges with weights (complete graph, same as exhaustive)
        for i in all_nodes:
            for j in all_nodes:
                if i < j:  # Avoid duplicates
                    weight = random.randint(1, 10)
                    network_graph.add_edge(i, j, weight=weight)
                    network_graph.nodes[i]["links"] = network_graph.nodes[i]["links"] + 1
                    network_graph.nodes[j]["links"] = network_graph.nodes[j]["links"] + 1

    # Run SA algorithm with provided graph (positional arguments)
    best_solution, initial_cost, final_cost, initial_avg_weight, final_avg_weight = sa.run_sa_algorithm(
        network_graph,  # positional arg 1
        constrained_nodes,  # positional arg 2
        mandatory_power_nodes,  # positional arg 3
        discretionary_power_nodes,  # positional arg 4
        initial_temperature,  # positional arg 5
        k_factor,  # positional arg 6
        'weight'  # weight_attr - positional arg 7
    )

    # ⏱️ STOP TIMING
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Convert solution to standard format
    result = {
        'best_tree': best_solution,
        'execution_time': elapsed_time,
        'num_nodes': best_solution.number_of_nodes() if best_solution else 0,
        'num_edges': best_solution.number_of_edges() if best_solution else 0,

        # Additional metrics specific to SA
        'initial_cost': initial_cost,
        'final_cost': final_cost,
        'initial_avg_weight': initial_avg_weight,
        'final_avg_weight': final_avg_weight,
        'cost_improvement': initial_cost - final_cost,
        'weight_improvement': initial_avg_weight - final_avg_weight,

        # SA uses ACC + AOC
        'acc': sa.calculate_acc(best_solution, weight_attr='weight'),
        'aoc': sa.calculate_aoc(best_solution)
    }

    return result


if __name__ == "__main__":
    # Test the wrapper
    print("Testing Simulated Annealing Algorithm Wrapper")
    print("=" * 60)

    from config_loader import load_config
    from graph_generator import generate_complete_config

    # Load configuration
    config = load_config('config.json')
    graph_config = generate_complete_config(config)

    # Algorithm configuration
    algorithm_config = {
        'seed': 42,
        'initial_temperature': 120,
        'k_factor': 12
    }

    print(f"\nGraph Configuration:")
    print(f"  Weak (constrained) nodes: {graph_config['weak_nodes']}")
    print(f"  Mandatory nodes: {graph_config['power_nodes_mandatory']}")
    print(f"  Discretionary nodes: {graph_config['power_nodes_discretionary']}")
    print(f"  Capacities: {graph_config['capacities']}")
    print(f"  Seed: {algorithm_config['seed']}")
    print(f"  Initial temperature: {algorithm_config['initial_temperature']}")
    print(f"  K factor: {algorithm_config['k_factor']}")

    # Run SA algorithm
    print(f"\nRunning Simulated Annealing algorithm...")
    result = run_sa_algorithm(graph_config, algorithm_config)

    # Display results
    print(f"\n{'=' * 60}")
    print("SIMULATED ANNEALING RESULTS")
    print(f"{'=' * 60}")
    print(f"✓ Execution time: {result['execution_time']:.4f}s")
    print(f"  Nodes in solution: {result['num_nodes']}")
    print(f"  Edges in solution: {result['num_edges']}")
    print(f"\n  Cost Evolution:")
    print(f"    Initial cost: {result['initial_cost']:.6f}")
    print(f"    Final cost: {result['final_cost']:.6f}")
    print(f"    Improvement: {result['cost_improvement']:.6f} ({(result['cost_improvement']/result['initial_cost']*100 if result['initial_cost'] > 0 else 0):.1f}%)")
    print(f"\n  Average Weight Evolution:")
    print(f"    Initial: {result['initial_avg_weight']:.2f}")
    print(f"    Final: {result['final_avg_weight']:.2f}")
    print(f"    Improvement: {result['weight_improvement']:.2f} ({(result['weight_improvement']/result['initial_avg_weight']*100 if result['initial_avg_weight'] > 0 else 0):.1f}%)")
    print(f"\n  Cost Components:")
    print(f"    ACC: {result['acc']:.6f}")
    print(f"    AOC: {result['aoc']:.6f}")
    print(f"\n{'=' * 60}")
