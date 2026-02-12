# graph_generator.py
"""
Graph Generator - Creates graphs with specified node configurations
Can be used standalone or as a library
"""

import sys
import random


def generate_graph_config(num_nodes=None, weak_ratio=None, mandatory_ratio=None,
                         num_weak=None, num_mandatory=None, num_discretionary=None,
                         seed=None):
    """
    Generate graph configuration with node assignments.

    Can specify either:
    - num_nodes + ratios (weak_ratio, mandatory_ratio)
    - Absolute numbers (num_weak, num_mandatory, num_discretionary)

    Args:
        num_nodes: Total number of nodes
        weak_ratio: Ratio of weak nodes (0-1)
        mandatory_ratio: Ratio of mandatory power nodes (0-1)
        num_weak: Absolute number of weak nodes
        num_mandatory: Absolute number of mandatory power nodes
        num_discretionary: Absolute number of discretionary power nodes
        seed: Random seed for reproducibility

    Returns:
        dict with keys: weak_nodes, power_nodes_mandatory, power_nodes_discretionary,
                       num_nodes, num_weak, num_mandatory, num_discretionary
    """
    if seed is not None:
        random.seed(seed)

    # Validate input
    if num_nodes is not None and weak_ratio is not None and mandatory_ratio is not None:
        # Calculate from ratios
        num_weak_calc = int(weak_ratio * num_nodes)
        num_mandatory_calc = int(mandatory_ratio * num_nodes)
        num_discretionary_calc = num_nodes - num_weak_calc - num_mandatory_calc

        if num_discretionary_calc < 0:
            raise ValueError("Ratios sum to more than 1.0")

        num_weak = num_weak_calc
        num_mandatory = num_mandatory_calc
        num_discretionary = num_discretionary_calc

    elif num_weak is not None and num_mandatory is not None and num_discretionary is not None:
        # Use absolute numbers
        num_nodes = num_weak + num_mandatory + num_discretionary
    else:
        raise ValueError("Must specify either (num_nodes + ratios) or (absolute numbers)")

    # Validate special cases
    if num_weak == num_nodes:
        raise ValueError("Cannot have only weak nodes")

    # Create node ranges
    weak_nodes = list(range(1, num_weak + 1))
    power_nodes_mandatory = list(range(num_weak + 1, num_weak + num_mandatory + 1))
    power_nodes_discretionary = list(range(num_weak + num_mandatory + 1,
                                          num_weak + num_mandatory + num_discretionary + 1))

    return {
        'weak_nodes': weak_nodes,
        'power_nodes_mandatory': power_nodes_mandatory,
        'power_nodes_discretionary': power_nodes_discretionary,
        'num_nodes': num_nodes,
        'num_weak': num_weak,
        'num_mandatory': num_mandatory,
        'num_discretionary': num_discretionary
    }


def generate_capacities(nodes, capacity_config, seed=None):
    """
    Generate capacity dictionary for nodes with multiple options.

    Args:
        nodes: List or range of node IDs
        capacity_config: Dict with one of these formats:
            - {"default": 10} - all nodes get same capacity
            - {"default": 10, "custom": {2: 30, 3: 2}} - default + overrides
            - {"random": {"min": 5, "max": 20, "seed": 42}} - random in range
            - {"random": {...}, "custom": {2: 30}} - random + overrides
            - {1: 10, 2: 30, ...} - explicit for each node (old format)
        seed: Random seed for reproducibility (overridden by random.seed if present)

    Returns:
        dict with node_id: capacity
    """
    if seed is not None:
        random.seed(seed)

    # Check if it's the old explicit format (all keys are node IDs)
    if all(isinstance(k, (int, str)) and k not in ['default', 'custom', 'random', 'comment']
           for k in capacity_config.keys()):
        # Old format: explicit capacities for each node
        return {int(k): v for k, v in capacity_config.items()}

    # New format with default/random/custom
    capacities = {}

    # Handle random generation
    if 'random' in capacity_config:
        random_config = capacity_config['random']
        min_cap = random_config.get('min', 1)
        max_cap = random_config.get('max', 10)
        random_seed = random_config.get('seed', seed)

        if random_seed is not None:
            random.seed(random_seed)

        # Generate random capacities for all nodes
        for node in nodes:
            capacities[node] = random.randint(min_cap, max_cap)

    # Handle default value
    elif 'default' in capacity_config:
        default_capacity = capacity_config['default']
        capacities = {node: default_capacity for node in nodes}

    else:
        # No default or random specified, use default of 10
        capacities = {node: 10 for node in nodes}

    # Apply custom overrides if present
    if 'custom' in capacity_config:
        custom = capacity_config['custom']
        # Convert string keys to int
        custom_int = {int(k): v for k, v in custom.items()}
        capacities.update(custom_int)

    return capacities


def generate_complete_config(config_params):
    """
    Generate complete configuration from parameters.

    Args:
        config_params: dict with graph_parameters, capacities, etc.

    Returns:
        dict with all necessary configuration
    """
    graph_params = config_params['graph_parameters']

    # Generate node configuration
    if 'num_nodes' in graph_params:
        graph_config = generate_graph_config(
            num_nodes=graph_params['num_nodes'],
            weak_ratio=graph_params.get('weak_ratio', 0.4),
            mandatory_ratio=graph_params.get('mandatory_ratio', 0.2),
            seed=graph_params.get('seed')
        )
    else:
        graph_config = generate_graph_config(
            num_weak=graph_params['num_weak'],
            num_mandatory=graph_params['num_mandatory'],
            num_discretionary=graph_params['num_discretionary'],
            seed=graph_params.get('seed')
        )

    # Generate capacities
    all_nodes = (graph_config['weak_nodes'] +
                graph_config['power_nodes_mandatory'] +
                graph_config['power_nodes_discretionary'])

    capacities_config = config_params.get('capacities', {})
    if isinstance(capacities_config, dict) and 'default' in capacities_config:
        # Handle default + custom format
        capacities = generate_capacities(
            all_nodes,
            default_capacity=capacities_config['default'],
            custom_capacities=capacities_config.get('custom', {})
        )
    else:
        # Direct capacity mapping (convert string keys to int)
        capacities = {int(k): v for k, v in capacities_config.items()}

    # Combine configurations
    result = {
        'weak_nodes': graph_config['weak_nodes'],
        'power_nodes_mandatory': graph_config['power_nodes_mandatory'],
        'power_nodes_discretionary': graph_config['power_nodes_discretionary'],
        'capacities': capacities,
        'num_nodes': graph_config['num_nodes']
    }

    return result


if __name__ == "__main__":
    # Standalone mode - demonstrate graph generation
    if len(sys.argv) > 1 and sys.argv[1] == '--config':
        from config_loader import load_config
        config = load_config(sys.argv[2] if len(sys.argv) > 2 else 'config.json')

        graph_config = generate_complete_config(config)

        print("Generated Graph Configuration:")
        print(f"  Total nodes: {graph_config['num_nodes']}")
        print(f"  Weak nodes: {graph_config['weak_nodes']}")
        print(f"  Mandatory power nodes: {graph_config['power_nodes_mandatory']}")
        print(f"  Discretionary power nodes: {graph_config['power_nodes_discretionary']}")
        print(f"  Capacities: {graph_config['capacities']}")
    else:
        # Demo with default values
        print("Demo: Generating graph configuration with default parameters")
        config = generate_graph_config(
            num_nodes=8,
            weak_ratio=0.4,
            mandatory_ratio=0.2,
            seed=42
        )

        all_nodes = (config['weak_nodes'] +
                    config['power_nodes_mandatory'] +
                    config['power_nodes_discretionary'])

        capacities = generate_capacities(all_nodes, default_capacity=10,
                                        custom_capacities={2: 30, 3: 2, 4: 1})

        print(f"Total nodes: {config['num_nodes']}")
        print(f"Weak nodes ({config['num_weak']}): {config['weak_nodes']}")
        print(f"Mandatory power nodes ({config['num_mandatory']}): {config['power_nodes_mandatory']}")
        print(f"Discretionary power nodes ({config['num_discretionary']}): {config['power_nodes_discretionary']}")
        print(f"Capacities: {capacities}")
