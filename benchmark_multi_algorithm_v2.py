"""
benchmark_multi_algorithm.py - FIXED VERSION

Multi-Algorithm Incremental Benchmark System
- Compare Exhaustive, Greedy, and Simulated Annealing
- Test 3 discretionary ratios: 10%, 30%, 50%
- Quality metrics for 50% discretionary (vs exhaustive ground truth)
- Incremental: --seeds, --resume --add-seeds, --extend-nodes

FIX: All algorithms now evaluated using EXHAUSTIVE cost function for fair comparison

Configurations:
  1-3:   10% discretionary (exhaustive, greedy, SA)
  4-6:   30% discretionary (exhaustive, greedy, SA)
  7-9:   50% discretionary (exhaustive, greedy, SA) + QUALITY METRICS

Usage:
    python3 benchmark_multi_algorithm.py <config_num> --seeds <N>
    python3 benchmark_multi_algorithm.py <config_num> --resume --add-seeds <M>
    python3 benchmark_multi_algorithm.py <config_num> --extend-nodes <nodes>

Output directory: multi_algorithm_results/
"""

import sys
import json
import time
import os
import pickle
import fcntl
import numpy as np
from datetime import datetime
from graph_generator import generate_graph_config
from benchmark import create_graph_with_strategy

# Import algorithms
from tree_optimizer_memory_optimized import run_algorithm as run_exhaustive
from greedy_algorithm_wrapper import run_greedy_algorithm
from sa_algorithm_wrapper import run_sa_algorithm


# ======================== CONFIGURATION ========================

# Test parameters
NUM_NODES_RANGE = [5, 6, 7, 8, 9]

# Discretionary ratios to test
DISCRETIONARY_RATIOS = [0.1, 0.3, 0.5]  # 10%, 30%, 50%

# Algorithms
ALGORITHMS = ['exhaustive', 'greedy', 'sa']

# Base configuration (same for all)
BASE_CONFIG = {
    'weak_ratio': 0.4,
    'mandatory_ratio': 0.3,
    'capacity': {
        'name': 'high_contrast',
        'type': 'scaled',
        'mandatory': {'min_mult': 0.8, 'max_mult': 1.2},
        'discretionary': {'min_mult': 6.0, 'max_mult': 10.0}
    },
    'weight_strategy': {
        'name': 'strong_favor',
        'default_range': (10, 30),
        'discretionary_range': (1, 3)
    }
}

# SA parameters
SA_CONFIG = {
    'initial_temperature': 120,
    'k_factor': 12
}

# Greedy parameters
GREEDY_CONFIG = {
    'alpha': 0.5
}

# Output directory
OUTPUT_DIR = 'multi_algorithm_results'
SOLUTIONS_DIR = os.path.join(OUTPUT_DIR, 'solutions')

# ================================================================


def get_config_list():
    """Generate list of all configurations"""
    configs = []
    config_num = 0
    
    for disc_ratio in DISCRETIONARY_RATIOS:
        for algorithm in ALGORITHMS:
            config_num += 1
            config_name = f"{int(disc_ratio*100)}pct_{algorithm}"
            
            configs.append({
                'num': config_num,
                'name': config_name,
                'discretionary_ratio': disc_ratio,
                'algorithm': algorithm,
                'has_quality': (disc_ratio == 0.5 and algorithm != 'exhaustive')
            })
    
    return configs


class ConfigLock:
    """File-based lock using PID"""
    def __init__(self, config_name, output_dir):
        self.lock_file_path = os.path.join(output_dir, f'.lock_{config_name}')
        self.pid = os.getpid()
    
    def acquire(self):
        """Acquire lock with PID check"""
        if os.path.exists(self.lock_file_path):
            try:
                with open(self.lock_file_path, 'r') as f:
                    old_pid = int(f.read().strip())
                
                try:
                    os.kill(old_pid, 0)
                    return False
                except OSError:
                    os.remove(self.lock_file_path)
            except:
                pass
        
        with open(self.lock_file_path, 'w') as f:
            f.write(str(self.pid))
        return True
    
    def release(self):
        """Release lock"""
        try:
            with open(self.lock_file_path, 'r') as f:
                lock_pid = int(f.read().strip())
            
            if lock_pid == self.pid:
                os.remove(self.lock_file_path)
        except:
            pass


def calculate_solution_metrics(best_tree, capacities, power_nodes_mandatory, 
                               power_nodes_discretionary, weak_nodes, global_max_weight):
    """
    FIXED: Calculate metrics using EXHAUSTIVE cost function for fair comparison
    
    CRITICAL FIX: Uses global_max_weight from the original graph, not from the solution tree.
    This ensures all algorithms are normalized against the same reference.
    
    This ensures all algorithms (exhaustive, greedy, SA) are evaluated with the same metric,
    making results directly comparable.
    """
    if best_tree is None or best_tree.number_of_edges() == 0:
        return {
            'edge_cost': float('inf'),
            'degree_cost': float('inf'),
            'total_cost': float('inf'),
            'discretionary_used': 0,
            'num_nodes': 0,
            'num_edges': 0,
            'solution_nodes': [],
            'solution_edges': []
        }
    
    edges_with_weights = [(edge, best_tree.get_edge_data(edge[0], edge[1])) 
                         for edge in best_tree.edges()]
    
    # ===== EDGE COST: identico a exhaustive =====
    # CRITICAL: Use global_max_weight from original graph, not from solution tree
    max_weight = global_max_weight
    total_weight = sum(data['weight'] for edge, data in edges_with_weights)
    
    max_nodes = len(weak_nodes) + len(power_nodes_mandatory) + len(power_nodes_discretionary)
    max_possible_edges = max_nodes - 1
    max_possible_weight = max_weight * max_possible_edges
    
    edge_cost = total_weight / max_possible_weight if max_possible_weight > 0 else 0
    
    # ===== DEGREE COST: identico a exhaustive =====
    all_power = set(list(power_nodes_mandatory) + list(power_nodes_discretionary))
    power_in_tree = [n for n in best_tree.nodes() if n in all_power]
    
    degree_sum = sum(best_tree.degree(n) / capacities[n] 
                    for n in power_in_tree if n in capacities)
    degree_cost_raw = degree_sum / len(power_in_tree) if power_in_tree else 0
    
    # Normalizza per caso peggiore
    min_capacity = min(capacities[n] for n in all_power if n in capacities) if capacities and all_power else 1
    max_degree_possible = max_nodes - 1
    max_degree_cost_per_node = max_degree_possible / min_capacity if min_capacity > 0 else 1
    
    degree_cost = degree_cost_raw / max_degree_cost_per_node if max_degree_cost_per_node > 0 else 0
    
    # ===== TOTAL COST: identico a exhaustive =====
    total_cost = edge_cost + degree_cost
    
    # Count discretionary used
    discretionary_used = sum(1 for n in best_tree.nodes() if n in power_nodes_discretionary)
    
    return {
        'edge_cost': edge_cost,
        'degree_cost': degree_cost,
        'total_cost': total_cost,
        'discretionary_used': discretionary_used,
        'num_nodes': best_tree.number_of_nodes(),
        'num_edges': best_tree.number_of_edges(),
        'solution_nodes': list(best_tree.nodes()),
        'solution_edges': [list(edge) for edge in best_tree.edges()]
    }


def save_solution_to_file(num_nodes, seed, solution_nodes, solution_edges, cost, algorithm, disc_ratio):
    """Save solution to JSON file"""
    solutions_dir = os.path.join(SOLUTIONS_DIR, f"{int(disc_ratio*100)}pct_{algorithm}")
    os.makedirs(solutions_dir, exist_ok=True)
    
    filepath = os.path.join(solutions_dir, f'solutions_nodes_{num_nodes}.json')
    
    # Load existing or create new
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            solutions = json.load(f)
    else:
        solutions = {}
    
    # Add this solution
    solutions[str(seed)] = {
        'nodes': solution_nodes,
        'edges': solution_edges,
        'cost': cost
    }
    
    # Save
    with open(filepath, 'w') as f:
        json.dump(solutions, f, indent=2)


def load_exhaustive_solution(num_nodes, seed, disc_ratio):
    """Load exhaustive solution from file"""
    filepath = os.path.join(SOLUTIONS_DIR, f"{int(disc_ratio*100)}pct_exhaustive", 
                           f'solutions_nodes_{num_nodes}.json')
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        solutions = json.load(f)
    
    return solutions.get(str(seed))


def calculate_quality_metrics(solution_algo, solution_exhaustive):
    """
    Calculate quality metrics comparing algorithm solution vs exhaustive
    
    Returns dict with:
        - same_solution: bool
        - cost_diff: absolute difference
        - cost_ratio: algo_cost / exhaustive_cost
        - node_overlap: fraction of common nodes
        - edge_overlap: fraction of common edges
    """
    if solution_exhaustive is None:
        return None
    
    # Extract data
    nodes_ex = set(solution_exhaustive['nodes'])
    edges_ex = set(tuple(sorted(e)) for e in solution_exhaustive['edges'])
    cost_ex = solution_exhaustive['cost']
    
    nodes_algo = set(solution_algo['nodes'])
    edges_algo = set(tuple(sorted(e)) for e in solution_algo['edges'])
    cost_algo = solution_algo['cost']
    
    # Calculate metrics
    same_solution = (nodes_ex == nodes_algo and edges_ex == edges_algo)
    
    cost_diff = abs(cost_algo - cost_ex)
    cost_ratio = cost_algo / cost_ex if cost_ex > 0 else float('inf')
    
    common_nodes = nodes_ex & nodes_algo
    node_overlap = len(common_nodes) / len(nodes_ex) if len(nodes_ex) > 0 else 0
    
    common_edges = edges_ex & edges_algo
    edge_overlap = len(common_edges) / len(edges_ex) if len(edges_ex) > 0 else 0
    
    return {
        'same_solution': same_solution,
        'cost_diff': cost_diff,
        'cost_ratio': cost_ratio,
        'node_overlap': node_overlap,
        'edge_overlap': edge_overlap
    }


# def run_single_test(num_nodes, disc_ratio, algorithm, seed, cap_config, weight_strategy):
#     """Run a single test"""
    
#     # Calculate base nodes
#     num_weak_base = int(num_nodes * BASE_CONFIG['weak_ratio'])
#     num_mandatory_base = int(num_nodes * BASE_CONFIG['mandatory_ratio'])
#     num_discretionary = int(num_nodes * disc_ratio)
    
#     graph_config_dict = generate_graph_config(
#         num_weak=num_weak_base,
#         num_mandatory=num_mandatory_base,
#         num_discretionary=num_discretionary,
#         seed=seed
#     )
    
#     all_nodes = (graph_config_dict['weak_nodes'] +
#                 graph_config_dict['power_nodes_mandatory'] +
#                 graph_config_dict['power_nodes_discretionary'])
    
#     # Generate capacities
#     import random
#     rng = random.Random(seed)
#     capacities = {}
    
#     for node in graph_config_dict['weak_nodes']:
#         capacities[node] = float('inf')
    
#     actual_num_nodes = len(all_nodes)
    
#     mand_min = max(1, int(actual_num_nodes * cap_config['mandatory']['min_mult']))
#     mand_max = max(mand_min + 1, int(actual_num_nodes * cap_config['mandatory']['max_mult']))
#     for node in graph_config_dict['power_nodes_mandatory']:
#         capacities[node] = rng.randint(mand_min, mand_max)
    
#     disc_min = max(1, int(actual_num_nodes * cap_config['discretionary']['min_mult']))
#     disc_max = max(disc_min + 1, int(actual_num_nodes * cap_config['discretionary']['max_mult']))
#     for node in graph_config_dict['power_nodes_discretionary']:
#         capacities[node] = rng.randint(disc_min, disc_max)
    
#     # Create graph
#     shared_graph = create_graph_with_strategy(
#         graph_config_dict['weak_nodes'],
#         graph_config_dict['power_nodes_mandatory'],
#         graph_config_dict['power_nodes_discretionary'],
#         capacities,
#         weight_strategy,
#         seed
#     )
    
#     # CRITICAL FIX: Calculate global_max_weight from the original graph
#     global_max_weight = max(shared_graph[u][v]['weight'] for u, v in shared_graph.edges())
    
#     graph_config = {
#         'weak_nodes': graph_config_dict['weak_nodes'],
#         'power_nodes_mandatory': graph_config_dict['power_nodes_mandatory'],
#         'power_nodes_discretionary': graph_config_dict['power_nodes_discretionary'],
#         'capacities': capacities
#     }
    
#     #debug_config = {'verbose': True, 'save_plots': False}
#     debug_config = {'verbose': False, 'save_plots': False}
#     start_time = time.time()
    
#     try:
#         if algorithm == 'exhaustive':
#             result = run_exhaustive(
#                 graph_config=graph_config,
#                 algorithm_config={'seed': seed},
#                 debug_config=debug_config,
#                 pre_built_graph=shared_graph
#             )
#             best_tree = result['best_tree']
            
#         elif algorithm == 'greedy':
#             result = run_greedy_algorithm(
#                 graph_config=graph_config,
#                 algorithm_config={'seed': seed, 'alpha': GREEDY_CONFIG['alpha']},
#                 pre_built_graph=shared_graph
#             )
#             best_tree = result['best_tree']
            
#         elif algorithm == 'sa':
#             result = run_sa_algorithm(
#                 graph_config=graph_config,
#                 algorithm_config={
#                     'seed': seed,
#                     'initial_temperature': SA_CONFIG['initial_temperature'],
#                     'k_factor': SA_CONFIG['k_factor']
#                 },
#                 pre_built_graph=shared_graph
#             )
#             best_tree = result['best_tree']
        
#         else:
#             raise ValueError(f"Unknown algorithm: {algorithm}")
        
#         execution_time = time.time() - start_time
        
#         # FIXED: Pass global_max_weight to calculate_solution_metrics
#         metrics = calculate_solution_metrics(
#             best_tree,
#             capacities,
#             graph_config_dict['power_nodes_mandatory'],
#             graph_config_dict['power_nodes_discretionary'],
#             graph_config_dict['weak_nodes'],
#             global_max_weight  # ← ADDED
#         )
        
#         metrics['execution_time'] = execution_time
#         metrics['success'] = True
        
#         # Save solution if exhaustive (for quality comparison)
#         if algorithm == 'exhaustive':
#             save_solution_to_file(
#                 num_nodes, seed,
#                 metrics['solution_nodes'],
#                 metrics['solution_edges'],
#                 metrics['total_cost'],
#                 algorithm, disc_ratio
#             )
        
#     except Exception as e:
#         execution_time = time.time() - start_time
#         metrics = {
#             'edge_cost': float('inf'),
#             'degree_cost': float('inf'),
#             'total_cost': float('inf'),
#             'discretionary_used': 0,
#             'num_nodes': 0,
#             'num_edges': 0,
#             'solution_nodes': [],
#             'solution_edges': [],
#             'execution_time': execution_time,
#             'success': False,
#             'error': str(e)
#         }
    
#     return metrics


def run_single_test(num_nodes, disc_ratio, algorithm, seed, cap_config, weight_strategy):
    """
    Run a single test
    
    FIXED: Discretionary ratio is FIXED percentage of total nodes,
           weak and mandatory split the REMAINING nodes proportionally
    """
    
    # FIXED: disc_ratio è percentuale FISSA del totale
    num_discretionary = int(num_nodes * disc_ratio)
    
    # Il rimanente si divide tra weak e mandatory secondo BASE_CONFIG
    remaining_nodes = num_nodes - num_discretionary
    
    weak_ratio = BASE_CONFIG['weak_ratio']          # 0.4
    mandatory_ratio = BASE_CONFIG['mandatory_ratio']  # 0.3
    total_base_ratio = weak_ratio + mandatory_ratio    # 0.7
    
    # Weak e mandatory sono percentuali RELATIVE del rimanente
    if total_base_ratio > 0:
        num_weak_base = int(remaining_nodes * (weak_ratio / total_base_ratio))
        num_mandatory_base = remaining_nodes - num_weak_base
    else:
        # Fallback se entrambi i ratio sono 0
        num_weak_base = 0
        num_mandatory_base = remaining_nodes
    
    graph_config_dict = generate_graph_config(
        num_weak=num_weak_base,
        num_mandatory=num_mandatory_base,
        num_discretionary=num_discretionary,
        seed=seed
    )
    
    all_nodes = (graph_config_dict['weak_nodes'] +
                graph_config_dict['power_nodes_mandatory'] +
                graph_config_dict['power_nodes_discretionary'])
    
    # Generate capacities
    import random
    rng = random.Random(seed)
    capacities = {}
    
    for node in graph_config_dict['weak_nodes']:
        capacities[node] = float('inf')
    
    actual_num_nodes = len(all_nodes)
    
    mand_min = max(1, int(actual_num_nodes * cap_config['mandatory']['min_mult']))
    mand_max = max(mand_min + 1, int(actual_num_nodes * cap_config['mandatory']['max_mult']))
    for node in graph_config_dict['power_nodes_mandatory']:
        capacities[node] = rng.randint(mand_min, mand_max)
    
    disc_min = max(1, int(actual_num_nodes * cap_config['discretionary']['min_mult']))
    disc_max = max(disc_min + 1, int(actual_num_nodes * cap_config['discretionary']['max_mult']))
    for node in graph_config_dict['power_nodes_discretionary']:
        capacities[node] = rng.randint(disc_min, disc_max)
    
    # Create graph
    shared_graph = create_graph_with_strategy(
        graph_config_dict['weak_nodes'],
        graph_config_dict['power_nodes_mandatory'],
        graph_config_dict['power_nodes_discretionary'],
        capacities,
        weight_strategy,
        seed
    )
    
    # CRITICAL FIX: Calculate global_max_weight from the original graph
    global_max_weight = max(shared_graph[u][v]['weight'] for u, v in shared_graph.edges())
    
    graph_config = {
        'weak_nodes': graph_config_dict['weak_nodes'],
        'power_nodes_mandatory': graph_config_dict['power_nodes_mandatory'],
        'power_nodes_discretionary': graph_config_dict['power_nodes_discretionary'],
        'capacities': capacities
    }
    
    debug_config = {'verbose': False, 'save_plots': False}
    start_time = time.time()
    
    try:
        if algorithm == 'exhaustive':
            result = run_exhaustive(
                graph_config=graph_config,
                algorithm_config={'seed': seed},
                debug_config=debug_config,
                pre_built_graph=shared_graph
            )
            best_tree = result['best_tree']
            
        elif algorithm == 'greedy':
            result = run_greedy_algorithm(
                graph_config=graph_config,
                algorithm_config={'seed': seed, 'alpha': GREEDY_CONFIG['alpha']},
                pre_built_graph=shared_graph
            )
            best_tree = result['best_tree']
            
        elif algorithm == 'sa':
            result = run_sa_algorithm(
                graph_config=graph_config,
                algorithm_config={
                    'seed': seed,
                    'initial_temperature': SA_CONFIG['initial_temperature'],
                    'k_factor': SA_CONFIG['k_factor']
                },
                pre_built_graph=shared_graph
            )
            best_tree = result['best_tree']
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        execution_time = time.time() - start_time
        
        # FIXED: Pass global_max_weight to calculate_solution_metrics
        metrics = calculate_solution_metrics(
            best_tree,
            capacities,
            graph_config_dict['power_nodes_mandatory'],
            graph_config_dict['power_nodes_discretionary'],
            graph_config_dict['weak_nodes'],
            global_max_weight  # ← ADDED
        )
        
        metrics['execution_time'] = execution_time
        metrics['success'] = True
        
        # Save solution if exhaustive (for quality comparison)
        if algorithm == 'exhaustive':
            save_solution_to_file(
                num_nodes, seed,
                metrics['solution_nodes'],
                metrics['solution_edges'],
                metrics['total_cost'],
                algorithm, disc_ratio
            )
        
    except Exception as e:
        execution_time = time.time() - start_time
        metrics = {
            'edge_cost': float('inf'),
            'degree_cost': float('inf'),
            'total_cost': float('inf'),
            'discretionary_used': 0,
            'num_nodes': 0,
            'num_edges': 0,
            'solution_nodes': [],
            'solution_edges': [],
            'execution_time': execution_time,
            'success': False,
            'error': str(e)
        }
    
    return metrics


def calculate_quality_for_seed(num_nodes, seed, solution_algo, disc_ratio):
    """Calculate quality metrics for this seed vs exhaustive"""
    
    # Load exhaustive solution
    sol_ex = load_exhaustive_solution(num_nodes, seed, disc_ratio)
    
    if sol_ex is None:
        return None
    
    # Prepare algorithm solution in same format
    sol_algo = {
        'nodes': solution_algo['solution_nodes'],
        'edges': solution_algo['solution_edges'],
        'cost': solution_algo['total_cost']
    }
    
    return calculate_quality_metrics(sol_algo, sol_ex)


def load_checkpoint(checkpoint_path):
    """Load existing checkpoint"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return None


def save_checkpoint(checkpoint_data, checkpoint_path):
    """Save checkpoint"""
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    # JSON version (without full solutions to keep it small)
    json_path = checkpoint_path.replace('.pkl', '.json')
    try:
        with open(json_path, 'w') as f:
            json_data = checkpoint_data.copy()
            # Only include summary of first 5 tests
            json_data['data'] = {
                str(k): [{'seed': t['seed'], 'success': t['metrics']['success'], 
                         'cost': t['metrics'].get('total_cost', 0)} for t in v[:5]]
                for k, v in checkpoint_data['data'].items()
            }
            json.dump(json_data, f, indent=2)
    except:
        pass


def calculate_statistics(data_list, include_quality=False):
    """Calculate statistics from list of test results"""
    if not data_list:
        return None
    
    successful = [t for t in data_list if t['metrics']['success']]
    if not successful:
        return {'n': 0, 'success_rate': 0.0}
    
    costs = [t['metrics']['total_cost'] for t in successful]
    times = [t['metrics']['execution_time'] for t in successful]
    discretionary_used = [t['metrics'].get('discretionary_used', 0) for t in successful]
    
    stats = {
        'n': len(successful),
        'n_total': len(data_list),
        'success_rate': len(successful) / len(data_list),
        'cost_mean': float(np.mean(costs)),
        'cost_std': float(np.std(costs)),
        'cost_median': float(np.median(costs)),
        'cost_min': float(np.min(costs)),
        'cost_max': float(np.max(costs)),
        'cost_cv': float(np.std(costs) / np.mean(costs) * 100) if np.mean(costs) > 0 else 0,
        'time_mean': float(np.mean(times)),
        'time_std': float(np.std(times)),
        'time_median': float(np.median(times)),
        'time_min': float(np.min(times)),
        'time_max': float(np.max(times)),
        'discretionary_used_mean': float(np.mean(discretionary_used)) if discretionary_used else 0.0
    }
    
    # Add quality metrics if available
    if include_quality:
        quality_list = [t.get('quality') for t in data_list if t.get('quality') is not None]
        
        if quality_list:
            stats['quality'] = {
                'match_rate': sum(1 for q in quality_list if q['same_solution']) / len(quality_list),
                'cost_diff_mean': float(np.mean([q['cost_diff'] for q in quality_list])),
                'cost_diff_std': float(np.std([q['cost_diff'] for q in quality_list])),
                'cost_ratio_mean': float(np.mean([q['cost_ratio'] for q in quality_list])),
                'cost_ratio_std': float(np.std([q['cost_ratio'] for q in quality_list])),
                'node_overlap_mean': float(np.mean([q['node_overlap'] for q in quality_list])),
                'edge_overlap_mean': float(np.mean([q['edge_overlap'] for q in quality_list]))
            }
    
    return stats


def run_configuration(config, target_seeds, resume=False):
    """Run one configuration with checkpoint support"""
    
    config_name = config['name']
    disc_ratio = config['discretionary_ratio']
    algorithm = config['algorithm']
    has_quality = config['has_quality']
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SOLUTIONS_DIR, exist_ok=True)
    
    # Acquire lock
    lock = ConfigLock(config_name, OUTPUT_DIR)
    if not lock.acquire():
        print(f"❌ Configuration '{config_name}' is already running!")
        print(f"   Remove lock: {OUTPUT_DIR}/.lock_{config_name}")
        sys.exit(1)
    
    try:
        checkpoint_path = os.path.join(OUTPUT_DIR, f'.checkpoint_{config_name}.pkl')
        
        # Load or create checkpoint
        if resume:
            checkpoint = load_checkpoint(checkpoint_path)
            if checkpoint is None:
                print(f"❌ No checkpoint found for '{config_name}'")
                sys.exit(1)
            
            start_seed = checkpoint['max_seed'] + 1
            print(f"\n{'='*100}")
            print(f"RESUMING: {config_name}")
            print(f"{'='*100}")
            print(f"From seed {start_seed} to {target_seeds}")
            
            if target_seeds <= checkpoint['max_seed']:
                print(f"✓ Already at {checkpoint['max_seed']} seeds")
                return checkpoint_path
        else:
            if os.path.exists(checkpoint_path):
                print(f"⚠️  Checkpoint exists for '{config_name}'")
                response = input(f"   Delete and start fresh? (yes/no): ")
                if response.lower() == 'yes':
                    os.remove(checkpoint_path)
                else:
                    sys.exit(1)
            
            start_seed = 1
            checkpoint = {
                'configuration': {
                    'discretionary_ratio': disc_ratio,
                    'algorithm': algorithm,
                    'has_quality': has_quality,
                    'capacity_config': BASE_CONFIG['capacity'],
                    'weight_strategy': BASE_CONFIG['weight_strategy']['name']
                },
                'max_seed': 0,
                'data': {num_nodes: [] for num_nodes in NUM_NODES_RANGE},
                'nodes_status': {num_nodes: {'complete_up_to_seed': 0, 'status': 'pending'} 
                                for num_nodes in NUM_NODES_RANGE},
                'statistics': {},
                'timestamp_started': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_execution_time': 0.0
            }
            
            print(f"\n{'='*100}")
            print(f"NEW RUN: {config_name}")
            print(f"{'='*100}")
            print(f"Target seeds: {target_seeds}")
        
        print(f"Discretionary: {disc_ratio*100:.0f}%")
        print(f"Algorithm: {algorithm}")
        print(f"Quality metrics: {'YES' if has_quality else 'NO'}")
        print(f"Cost function: EXHAUSTIVE (normalized edge+degree)")
        print(f"{'='*100}\n")
        
        run_start_time = time.time()
        
        # Run tests
        for seed in range(start_seed, target_seeds + 1):
            for num_nodes in NUM_NODES_RANGE:
                metrics = run_single_test(
                    num_nodes, disc_ratio, algorithm, seed,
                    BASE_CONFIG['capacity'], BASE_CONFIG['weight_strategy']
                )
                
                # Calculate quality if needed
                quality = None
                if has_quality and metrics['success']:
                    quality = calculate_quality_for_seed(
                        num_nodes, seed,
                        metrics,
                        disc_ratio
                    )
                
                checkpoint['data'][num_nodes].append({
                    'seed': seed,
                    'metrics': metrics,
                    'quality': quality
                })
                
                checkpoint['nodes_status'][num_nodes]['complete_up_to_seed'] = seed
                if seed == target_seeds:
                    checkpoint['nodes_status'][num_nodes]['status'] = 'complete'
            
            checkpoint['max_seed'] = seed
            checkpoint['timestamp_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            checkpoint['total_execution_time'] = time.time() - run_start_time
            
            # Save checkpoint every 10 seeds
            if seed % 10 == 0 or seed == target_seeds:
                save_checkpoint(checkpoint, checkpoint_path)
                
                print(f"\n  Checkpoint at seed {seed}/{target_seeds}")
                for num_nodes in NUM_NODES_RANGE:
                    stats = calculate_statistics(checkpoint['data'][num_nodes], has_quality)
                    if stats and stats['n'] > 0:
                        status = "✓" if stats['cost_cv'] < 10 else "⚠️"
                        time_info = f"time={stats['time_mean']:.4f}s"
                        
                        if has_quality and 'quality' in stats:
                            quality_info = f", match={stats['quality']['match_rate']*100:.0f}%"
                        else:
                            quality_info = ""
                        
                        print(f"    Nodes={num_nodes}: {time_info}, "
                              f"cost={stats['cost_mean']:.4f}±{stats['cost_std']:.4f}{quality_info} {status}")
            
            if seed % 50 == 0:
                elapsed = time.time() - run_start_time
                avg_time = elapsed / (seed - start_seed + 1)
                remaining = (target_seeds - seed) * avg_time
                print(f"  Progress: {seed}/{target_seeds} - ETA: {remaining/60:.1f}min")
        
        # Final statistics
        print(f"\n{'='*100}")
        print(f"COMPLETE: {config_name}")
        print(f"{'='*100}")
        
        checkpoint['statistics'] = {}
        for num_nodes in NUM_NODES_RANGE:
            stats = calculate_statistics(checkpoint['data'][num_nodes], has_quality)
            checkpoint['statistics'][num_nodes] = stats
            
            if stats and stats['n'] > 0:
                status = "✓ STABLE" if stats['cost_cv'] < 10 else "⚠️  NEEDS MORE"
                
                info = f"time={stats['time_mean']:.4f}s, cost={stats['cost_mean']:.4f}±{stats['cost_std']:.4f}"
                
                if has_quality and 'quality' in stats:
                    q = stats['quality']
                    info += f", match={q['match_rate']*100:.0f}%, ratio={q['cost_ratio_mean']:.3f}"
                
                print(f"Nodes={num_nodes}: {info} {status}")
        
        print(f"\nTotal time: {checkpoint['total_execution_time']/60:.1f}min")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*100}\n")
        
        save_checkpoint(checkpoint, checkpoint_path)
        return checkpoint_path
        
    finally:
        lock.release()


def extend_nodes_in_checkpoint(checkpoint_path, nodes_to_extend, config, force=False):
    """Extend checkpoint with new node sizes"""
    
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        print(f"❌ No checkpoint found")
        return None
    
    config_name = config['name']
    disc_ratio = config['discretionary_ratio']
    algorithm = config['algorithm']
    has_quality = config['has_quality']
    
    # Initialize nodes_status if needed
    if 'nodes_status' not in checkpoint:
        checkpoint['nodes_status'] = {}
        for num_nodes in checkpoint['data'].keys():
            checkpoint['nodes_status'][num_nodes] = {
                'complete_up_to_seed': checkpoint['max_seed'],
                'status': 'complete'
            }
    
    print(f"\n{'='*100}")
    print(f"EXTENDING: {config_name}")
    print(f"{'='*100}")
    print(f"Current nodes: {sorted(checkpoint['data'].keys())}")
    print(f"Max seed: {checkpoint['max_seed']}")
    
    extension_plan = []
    for num_nodes in nodes_to_extend:
        if num_nodes in checkpoint['data'] and not force:
            node_status = checkpoint['nodes_status'].get(num_nodes, {})
            if node_status.get('status') == 'complete':
                print(f"  ⚠️  Node {num_nodes}: already complete")
                continue
            elif node_status.get('status') == 'extending':
                resume_from = node_status.get('complete_up_to_seed', 0) + 1
                extension_plan.append((num_nodes, resume_from, checkpoint['max_seed']))
                print(f"  🔄 Node {num_nodes}: resume from seed {resume_from}")
            else:
                extension_plan.append((num_nodes, 1, checkpoint['max_seed']))
                print(f"  ♻️  Node {num_nodes}: recalculate")
        else:
            extension_plan.append((num_nodes, 1, checkpoint['max_seed']))
            print(f"  ➕ Node {num_nodes}: NEW")
    
    if not extension_plan:
        print(f"\n❌ No nodes to extend")
        return checkpoint_path
    
    print(f"\nTotal tests: {sum(end-start+1 for _, start, end in extension_plan)}")
    print(f"{'='*100}\n")
    
    run_start_time = time.time()
    
    for num_nodes, start_seed, end_seed in extension_plan:
        print(f"Processing node {num_nodes} (seeds {start_seed}→{end_seed})...")
        
        if num_nodes not in checkpoint['data']:
            checkpoint['data'][num_nodes] = []
        
        checkpoint['nodes_status'][num_nodes] = {
            'complete_up_to_seed': start_seed - 1,
            'status': 'extending'
        }
        
        for seed in range(start_seed, end_seed + 1):
            metrics = run_single_test(
                num_nodes, disc_ratio, algorithm, seed,
                BASE_CONFIG['capacity'], BASE_CONFIG['weight_strategy']
            )
            
            # Calculate quality if needed
            quality = None
            if has_quality and metrics['success']:
                quality = calculate_quality_for_seed(num_nodes, seed, metrics, disc_ratio)
            
            existing = next((t for t in checkpoint['data'][num_nodes] if t['seed'] == seed), None)
            
            if existing:
                existing['metrics'] = metrics
                existing['quality'] = quality
            else:
                checkpoint['data'][num_nodes].append({
                    'seed': seed,
                    'metrics': metrics,
                    'quality': quality
                })
            
            checkpoint['nodes_status'][num_nodes]['complete_up_to_seed'] = seed
            
            if seed % 10 == 0 or seed == end_seed:
                checkpoint['timestamp_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                save_checkpoint(checkpoint, checkpoint_path)
                print(f"  Checkpoint at seed {seed}/{end_seed}")
        
        checkpoint['nodes_status'][num_nodes]['status'] = 'complete'
        print(f"  ✓ Node {num_nodes} complete\n")
    
    # Update statistics
    checkpoint['statistics'] = {}
    for num_nodes in sorted(checkpoint['data'].keys()):
        checkpoint['statistics'][num_nodes] = calculate_statistics(
            checkpoint['data'][num_nodes], has_quality
        )
    
    save_checkpoint(checkpoint, checkpoint_path)
    
    print(f"✓ Extension complete: {checkpoint_path}\n")
    return checkpoint_path


def list_all_configurations():
    """List all configurations"""
    configs = get_config_list()
    
    print("\n" + "="*100)
    print("MULTI-ALGORITHM BENCHMARK CONFIGURATIONS")
    print("="*100 + "\n")
    
    for config in configs:
        print(f"{config['num']:2d}. {config['name']}")
        print(f"    Discretionary: {config['discretionary_ratio']*100:.0f}% | "
              f"Algorithm: {config['algorithm']}")
        
        if config['has_quality']:
            print(f"    ⭐ Includes QUALITY metrics (vs exhaustive)")
        
        checkpoint_path = os.path.join(OUTPUT_DIR, f".checkpoint_{config['name']}.pkl")
        if os.path.exists(checkpoint_path):
            cp = load_checkpoint(checkpoint_path)
            if cp:
                nodes = sorted(cp['data'].keys())
                print(f"    📂 Checkpoint: {cp['max_seed']} seeds, nodes {nodes}")
        print()
    
    print("="*100)
    print(f"Total: {len(configs)} configurations")
    print(f"Performance: 9 configs (3 ratios × 3 algorithms)")
    print(f"Quality: Only configs 8-9 (50% greedy/SA vs exhaustive)")
    print(f"Cost function: EXHAUSTIVE (normalized edge+degree) for ALL algorithms")
    print(f"Output: {OUTPUT_DIR}/")
    print("="*100 + "\n")
    
    return configs


def main():
    """Main launcher"""
    
    configs = list_all_configurations()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  python3 {sys.argv[0]} <config_num> --seeds <N>")
        print(f"  python3 {sys.argv[0]} <config_num> --resume --add-seeds <M>")
        print(f"  python3 {sys.argv[0]} <config_num> --extend-nodes <nodes>")
        print(f"\nExamples:")
        print(f"  python3 {sys.argv[0]} 1 --seeds 200")
        print(f"  python3 {sys.argv[0]} 7 --resume --add-seeds 100")
        print(f"  python3 {sys.argv[0]} 1 --extend-nodes 10,11")
        print(f"\nParallel:")
        print(f"  for i in {{1..9}}; do")
        print(f"    python3 {sys.argv[0]} $i --seeds 200 > {OUTPUT_DIR}/logs/config_$i.log 2>&1 &")
        print(f"  done")
        return
    
    try:
        config_num = int(sys.argv[1])
        if config_num < 1 or config_num > len(configs):
            print(f"❌ Config must be 1-{len(configs)}")
            return
    except ValueError:
        print(f"❌ Invalid config number")
        return
    
    config = configs[config_num - 1]
    
    # Handle extend-nodes
    if '--extend-nodes' in sys.argv:
        idx = sys.argv.index('--extend-nodes')
        if idx + 1 >= len(sys.argv):
            print("❌ --extend-nodes requires node list")
            return
        
        nodes_str = sys.argv[idx + 1]
        nodes_to_extend = [int(n.strip()) for n in nodes_str.split(',')]
        force = '--force' in sys.argv
        
        checkpoint_path = os.path.join(OUTPUT_DIR, f".checkpoint_{config['name']}.pkl")
        
        lock = ConfigLock(config['name'], OUTPUT_DIR)
        if not lock.acquire():
            print(f"❌ Config already running!")
            return
        
        try:
            extend_nodes_in_checkpoint(checkpoint_path, nodes_to_extend, config, force)
        finally:
            lock.release()
        return
    
    # Handle seeds
    resume = '--resume' in sys.argv
    
    if '--seeds' in sys.argv:
        idx = sys.argv.index('--seeds')
        target_seeds = int(sys.argv[idx + 1])
    elif '--add-seeds' in sys.argv:
        if not resume:
            print("❌ --add-seeds requires --resume")
            return
        idx = sys.argv.index('--add-seeds')
        checkpoint_path = os.path.join(OUTPUT_DIR, f".checkpoint_{config['name']}.pkl")
        cp = load_checkpoint(checkpoint_path)
        if cp is None:
            print("❌ No checkpoint found")
            return
        target_seeds = cp['max_seed'] + int(sys.argv[idx + 1])
    else:
        print("❌ Must specify --seeds or --add-seeds")
        return
    
    print(f"\n🚀 Running config {config_num}: {config['name']}")
    run_configuration(config, target_seeds, resume)


if __name__ == "__main__":
    main()