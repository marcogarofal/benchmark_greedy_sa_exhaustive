# benchmark.py
"""
Comprehensive benchmark for comparing Exhaustive, Greedy, and Simulated Annealing algorithms
Full factorial design: tests all combinations of num_nodes × ratios × capacities × weight_strategies × seeds
"""

from tree_optimizer_memory_optimized import run_algorithm as run_exhaustive
from greedy_algorithm_wrapper import run_greedy_algorithm
from sa_algorithm_wrapper import run_sa_algorithm
from graph_generator import generate_graph_config, generate_capacities
import json
import time
import datetime
import numpy as np
from collections import defaultdict
import networkx as nx
import random
import os


# ======================== CONFIGURABLE PARAMETERS ========================

# Number of seeds to test per configuration
SEEDS_PER_CONFIG = 100

# Configurations for NUM_NODES
NUM_NODES_CONFIGS = [5, 6, 7, 8, 9, 10, 11, 12]
#NUM_NODES_CONFIGS = [5, 6]

# Configurations for RATIOS (weak_ratio, mandatory_ratio)
RATIOS_CONFIGS = [
    (0.6, 0.2),  # Many weak, few mandatory, few discretionary
    (0.4, 0.2),  # Balanced (baseline)
    (0.2, 0.2),  # Few weak, many discretionary
]

# Configurations for CAPACITIES
CAPACITY_CONFIGS = [
    {'random': {'min': 1, 'max': 5, 'seed': None}},    # Low (high stress)
    {'default': 10},                                    # Medium (baseline)
    {'random': {'min': 20, 'max': 50, 'seed': None}},  # High (low stress)
]

# Edge weight strategies
EDGE_WEIGHT_STRATEGIES = [
    {
        'name': 'uniform',
        'description': 'All edges same weight range',
        'default_range': (1, 10),
        'discretionary_range': (1, 10)
    },
    {
        'name': 'favor_discretionary',
        'description': 'Discretionary edges have lower weights',
        'default_range': (1, 10),
        'discretionary_range': (1, 5)
    },
    {
        'name': 'strong_favor',
        'description': 'Strongly favor discretionary',
        'default_range': (5, 10),
        'discretionary_range': (1, 3)
    }
]

# Algorithm parameters
GREEDY_ALPHA = 0.5
SA_INITIAL_TEMPERATURE = 120
SA_K_FACTOR = 12

# Output directory
OUTPUT_DIR = 'benchmark_results'

# =========================================================================


def create_graph_with_strategy(weak_nodes, mandatory_nodes, discretionary_nodes,
                              capacities, weight_strategy, seed):
    """Create graph with specified weight strategy"""
    if seed is not None:
        random.seed(seed)

    G = nx.Graph()
    all_nodes = list(weak_nodes) + list(mandatory_nodes) + list(discretionary_nodes)
    discretionary_set = set(discretionary_nodes)

    # Add nodes
    for node in weak_nodes:
        G.add_node(node, node_type='weak', capacity=capacities.get(node, 1), links=0)
    for node in mandatory_nodes:
        G.add_node(node, node_type='power_mandatory', capacity=capacities.get(node, 10), links=0)
    for node in discretionary_nodes:
        G.add_node(node, node_type='power_discretionary', capacity=capacities.get(node, 10), links=0)

    # Add edges with strategy
    for i in all_nodes:
        for j in all_nodes:
            if i < j:
                if i in discretionary_set or j in discretionary_set:
                    min_w, max_w = weight_strategy['discretionary_range']
                else:
                    min_w, max_w = weight_strategy['default_range']

                weight = random.randint(min_w, max_w)
                G.add_edge(i, j, weight=weight)

    return G


def solutions_are_identical(result1, result2):
    """Check if two solutions are identical"""
    if not result1 or not result2:
        return False

    if not result1.get('best_tree') or not result2.get('best_tree'):
        return False

    nodes1 = set(result1['best_tree'].nodes())
    nodes2 = set(result2['best_tree'].nodes())
    edges1 = set(tuple(sorted(edge)) for edge in result1['best_tree'].edges())
    edges2 = set(tuple(sorted(edge)) for edge in result2['best_tree'].edges())

    return nodes1 == nodes2 and edges1 == edges2


def run_single_test_with_shared_graph(shared_graph, graph_config, config_params, seed, test_id):
    """Run all three algorithms on the SAME pre-built graph"""

    result = {
        'test_id': test_id,
        'config': config_params.copy(),
        'seed': seed,
        'graph_info': {
            'num_nodes': shared_graph.number_of_nodes(),
            'num_edges': shared_graph.number_of_edges(),
            'num_weak': len(graph_config['weak_nodes']),
            'num_mandatory': len(graph_config['power_nodes_mandatory']),
            'num_discretionary': len(graph_config['power_nodes_discretionary'])
        }
    }

    # Exhaustive - pass pre-built graph
    res_ex = None
    try:
        res_ex = run_exhaustive(
            graph_config,
            {'seed': seed},
            {'save_plots': False, 'verbose': False},
            'temp_plots',
            pre_built_graph=shared_graph
        )
        result['exhaustive'] = {
            'time': res_ex['execution_time'],
            'num_nodes': res_ex['num_nodes'],
            'num_edges': res_ex['num_edges'],
            'solution_nodes': list(res_ex['best_tree'].nodes()) if res_ex['best_tree'] else [],
            'solution_edges': [list(e) for e in res_ex['best_tree'].edges()] if res_ex['best_tree'] else []
        }
    except Exception as e:
        result['exhaustive'] = {'error': str(e)}
        print(f"    ✗ Exhaustive: {e}")

    # Greedy - pass pre-built graph
    res_gr = None
    try:
        res_gr = run_greedy_algorithm(
            graph_config,
            {'seed': seed, 'alpha': GREEDY_ALPHA},
            pre_built_graph=shared_graph
        )
        result['greedy'] = {
            'time': res_gr['execution_time'],
            'num_nodes': res_gr['num_nodes'],
            'num_edges': res_gr['num_edges'],
            'solution_nodes': list(res_gr['best_tree'].nodes()) if res_gr['best_tree'] else [],
            'solution_edges': [list(e) for e in res_gr['best_tree'].edges()] if res_gr['best_tree'] else [],
            'matches_exhaustive': solutions_are_identical(res_ex, res_gr) if res_ex is not None and 'error' not in result.get('exhaustive', {}) else False
        }
    except Exception as e:
        result['greedy'] = {'error': str(e)}
        print(f"    ✗ Greedy: {e}")

    # SA - pass pre-built graph
    res_sa = None
    try:
        res_sa = run_sa_algorithm(
            graph_config,
            {'seed': seed, 'initial_temperature': SA_INITIAL_TEMPERATURE, 'k_factor': SA_K_FACTOR},
            pre_built_graph=shared_graph
        )
        result['sa'] = {
            'time': res_sa['execution_time'],
            'num_nodes': res_sa['num_nodes'],
            'num_edges': res_sa['num_edges'],
            'solution_nodes': list(res_sa['best_tree'].nodes()) if res_sa['best_tree'] else [],
            'solution_edges': [list(e) for e in res_sa['best_tree'].edges()] if res_sa['best_tree'] else [],
            'matches_exhaustive': solutions_are_identical(res_ex, res_sa) if res_ex is not None and 'error' not in result.get('exhaustive', {}) else False
        }
    except Exception as e:
        result['sa'] = {'error': str(e)}
        print(f"    ✗ SA: {e}")

    # Brief output
    if test_id % 10 == 1 or 'error' in result.get('exhaustive', {}) or 'error' in result.get('greedy', {}) or 'error' in result.get('sa', {}):
        ex_time = result.get('exhaustive', {}).get('time', 0)
        gr_match = '✓' if result.get('greedy', {}).get('matches_exhaustive', False) else '✗'
        sa_match = '✓' if result.get('sa', {}).get('matches_exhaustive', False) else '✗'
        print(f"  #{test_id}: Ex={ex_time:.3f}s, Gr={gr_match}, SA={sa_match}")

    return result


def calculate_full_aggregation(all_results):
    """Calculate statistics aggregated by each dimension"""

    aggregated = {
        'by_num_nodes': defaultdict(lambda: {'times_ex': [], 'times_gr': [], 'times_sa': [],
                                             'gr_matches': 0, 'sa_matches': 0, 'total': 0}),
        'by_ratios': defaultdict(lambda: {'times_ex': [], 'times_gr': [], 'times_sa': [],
                                          'gr_matches': 0, 'sa_matches': 0, 'total': 0}),
        'by_capacities': defaultdict(lambda: {'times_ex': [], 'times_gr': [], 'times_sa': [],
                                              'gr_matches': 0, 'sa_matches': 0, 'total': 0}),
        'by_weight_strategy': defaultdict(lambda: {'times_ex': [], 'times_gr': [], 'times_sa': [],
                                                   'gr_matches': 0, 'sa_matches': 0, 'total': 0})
    }

    for test in all_results:
        # Extract keys
        nodes_key = str(test['config']['num_nodes'])
        ratios_key = f"w{int(test['config']['weak_ratio']*100)}_m{int(test['config']['mandatory_ratio']*100)}"
        cap_key = test['config']['capacity_config']
        strat_key = test['config']['weight_strategy']

        # Aggregate by each dimension
        for key, agg_dict in [
            (nodes_key, aggregated['by_num_nodes']),
            (ratios_key, aggregated['by_ratios']),
            (cap_key, aggregated['by_capacities']),
            (strat_key, aggregated['by_weight_strategy'])
        ]:
            agg = agg_dict[key]
            agg['total'] += 1

            if 'error' not in test.get('exhaustive', {}):
                agg['times_ex'].append(test['exhaustive']['time'])
            if 'error' not in test.get('greedy', {}):
                agg['times_gr'].append(test['greedy']['time'])
                if test['greedy'].get('matches_exhaustive', False):
                    agg['gr_matches'] += 1
            if 'error' not in test.get('sa', {}):
                agg['times_sa'].append(test['sa']['time'])
                if test['sa'].get('matches_exhaustive', False):
                    agg['sa_matches'] += 1

    # Calculate statistics
    final_agg = {}
    for dimension in ['by_num_nodes', 'by_ratios', 'by_capacities', 'by_weight_strategy']:
        final_agg[dimension] = {}
        for key, data in aggregated[dimension].items():
            stats = {'n_tests': data['total']}

            # Match rates
            stats['greedy_match_rate'] = data['gr_matches'] / data['total'] if data['total'] > 0 else 0
            stats['sa_match_rate'] = data['sa_matches'] / data['total'] if data['total'] > 0 else 0
            stats['greedy_match_count'] = data['gr_matches']
            stats['sa_match_count'] = data['sa_matches']

            # Time statistics
            for algo, times_key in [('exhaustive', 'times_ex'), ('greedy', 'times_gr'), ('sa', 'times_sa')]:
                times = data[times_key]
                if times:
                    stats[f'{algo}_time_mean'] = float(np.mean(times))
                    stats[f'{algo}_time_std'] = float(np.std(times))
                    stats[f'{algo}_time_median'] = float(np.median(times))
                    stats[f'{algo}_time_min'] = float(np.min(times))
                    stats[f'{algo}_time_max'] = float(np.max(times))

            final_agg[dimension][key] = stats

    return final_agg


def generate_comprehensive_report(all_results, aggregated, timestamp, total_time):
    """Generate comprehensive text report"""
    filename = f"{OUTPUT_DIR}/benchmark_report_{timestamp}.txt"

    with open(filename, 'w') as f:
        f.write("="*100 + "\n")
        f.write("COMPREHENSIVE ALGORITHM BENCHMARK REPORT\n")
        f.write("="*100 + "\n\n")

        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total tests: {len(all_results)}\n")
        f.write(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)\n")
        f.write(f"Seeds per config: {SEEDS_PER_CONFIG}\n\n")

        # Overall summary
        total_greedy_matches = sum(1 for r in all_results if r.get('greedy', {}).get('matches_exhaustive', False))
        total_sa_matches = sum(1 for r in all_results if r.get('sa', {}).get('matches_exhaustive', False))

        f.write("OVERALL SUMMARY\n")
        f.write("-"*50 + "\n")
        f.write(f"Greedy match rate: {total_greedy_matches}/{len(all_results)} ({total_greedy_matches/len(all_results)*100:.1f}%)\n")
        f.write(f"SA match rate: {total_sa_matches}/{len(all_results)} ({total_sa_matches/len(all_results)*100:.1f}%)\n\n")

        # By num nodes
        f.write("\n" + "="*100 + "\n")
        f.write("RESULTS BY NUMBER OF NODES\n")
        f.write("="*100 + "\n\n")

        f.write(f"{'Nodes':<8} {'Tests':<8} {'Greedy Match':<20} {'SA Match':<20} {'Exhaustive Time':<25} {'Greedy Time':<25}\n")
        f.write("-"*100 + "\n")

        for key in sorted(aggregated['by_num_nodes'].keys(), key=int):
            s = aggregated['by_num_nodes'][key]
            gr_match_str = f"{s['greedy_match_count']}/{s['n_tests']} ({s['greedy_match_rate']*100:.1f}%)"
            sa_match_str = f"{s['sa_match_count']}/{s['n_tests']} ({s['sa_match_rate']*100:.1f}%)"
            ex_time_str = f"{s.get('exhaustive_time_mean',0):.3f}±{s.get('exhaustive_time_std',0):.3f}s"
            gr_time_str = f"{s.get('greedy_time_mean',0):.4f}±{s.get('greedy_time_std',0):.4f}s"

            f.write(f"{key:<8} {s['n_tests']:<8} {gr_match_str:<20} {sa_match_str:<20} {ex_time_str:<25} {gr_time_str:<25}\n")

        # By ratios
        f.write("\n" + "="*100 + "\n")
        f.write("RESULTS BY NODE RATIOS\n")
        f.write("="*100 + "\n\n")

        f.write(f"{'Ratios':<15} {'Tests':<8} {'Greedy Match':<20} {'SA Match':<20}\n")
        f.write("-"*70 + "\n")

        for key in sorted(aggregated['by_ratios'].keys()):
            s = aggregated['by_ratios'][key]
            gr_match_str = f"{s['greedy_match_count']}/{s['n_tests']} ({s['greedy_match_rate']*100:.1f}%)"
            sa_match_str = f"{s['sa_match_count']}/{s['n_tests']} ({s['sa_match_rate']*100:.1f}%)"
            f.write(f"{key:<15} {s['n_tests']:<8} {gr_match_str:<20} {sa_match_str:<20}\n")

        # By capacities
        f.write("\n" + "="*100 + "\n")
        f.write("RESULTS BY CAPACITIES\n")
        f.write("="*100 + "\n\n")

        f.write(f"{'Capacity':<25} {'Tests':<8} {'Greedy Match':<20} {'SA Match':<20}\n")
        f.write("-"*80 + "\n")

        for key in sorted(aggregated['by_capacities'].keys()):
            s = aggregated['by_capacities'][key]
            gr_match_str = f"{s['greedy_match_count']}/{s['n_tests']} ({s['greedy_match_rate']*100:.1f}%)"
            sa_match_str = f"{s['sa_match_count']}/{s['n_tests']} ({s['sa_match_rate']*100:.1f}%)"
            f.write(f"{key:<25} {s['n_tests']:<8} {gr_match_str:<20} {sa_match_str:<20}\n")

        # By weight strategy
        f.write("\n" + "="*100 + "\n")
        f.write("RESULTS BY WEIGHT STRATEGY\n")
        f.write("="*100 + "\n\n")

        f.write(f"{'Strategy':<30} {'Tests':<8} {'Greedy Match':<20} {'SA Match':<20}\n")
        f.write("-"*85 + "\n")

        for key in sorted(aggregated['by_weight_strategy'].keys()):
            s = aggregated['by_weight_strategy'][key]
            gr_match_str = f"{s['greedy_match_count']}/{s['n_tests']} ({s['greedy_match_rate']*100:.1f}%)"
            sa_match_str = f"{s['sa_match_count']}/{s['n_tests']} ({s['sa_match_rate']*100:.1f}%)"
            f.write(f"{key:<30} {s['n_tests']:<8} {gr_match_str:<20} {sa_match_str:<20}\n")

        f.write("\n\nDetailed statistics available in aggregated_results JSON file.\n")

    print(f"\n📄 Report saved: {filename}")
    return filename


def run_benchmark():
    """Main benchmark function with full factorial design"""

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Calculate total tests
    total_configs = (len(NUM_NODES_CONFIGS) * len(RATIOS_CONFIGS) *
                    len(CAPACITY_CONFIGS) * len(EDGE_WEIGHT_STRATEGIES))
    total_tests = total_configs * SEEDS_PER_CONFIG

    print("="*100)
    print("COMPREHENSIVE ALGORITHM BENCHMARK - FULL FACTORIAL DESIGN")
    print("="*100)
    print(f"\nDimensions:")
    print(f"  Num nodes: {len(NUM_NODES_CONFIGS)} values")
    print(f"  Ratios: {len(RATIOS_CONFIGS)} combinations")
    print(f"  Capacities: {len(CAPACITY_CONFIGS)} configurations")
    print(f"  Weight strategies: {len(EDGE_WEIGHT_STRATEGIES)} strategies")
    print(f"  Seeds per config: {SEEDS_PER_CONFIG}")
    print(f"\n  Total configurations: {total_configs}")
    print(f"  TOTAL TESTS: {total_tests}")
    print(f"\nEstimated time: ~{total_tests * 0.5 / 3600:.1f} hours")

    input("\nPress ENTER to start benchmark...")

    all_results = []
    test_id = 0
    start_time_bench = time.time()

    # Full factorial loop
    print(f"\n{'='*100}")
    print("RUNNING FULL FACTORIAL TESTS")
    print(f"{'='*100}\n")

    for num_nodes in NUM_NODES_CONFIGS:
        for weak_ratio, mandatory_ratio in RATIOS_CONFIGS:
            for capacity_config in CAPACITY_CONFIGS:
                for weight_strategy in EDGE_WEIGHT_STRATEGIES:

                    # Configuration description
                    discr_pct = int((1 - weak_ratio - mandatory_ratio) * 100)
                    cap_desc = f"default_{capacity_config['default']}" if 'default' in capacity_config else f"rand_{capacity_config['random']['min']}-{capacity_config['random']['max']}"

                    print(f"\n{'─'*100}")
                    print(f"Config: {num_nodes}nodes, w{int(weak_ratio*100)}%_m{int(mandatory_ratio*100)}%_d{discr_pct}%, "
                          f"cap={cap_desc}, strategy={weight_strategy['name']}")
                    print(f"{'─'*100}")

                    for seed in range(1, SEEDS_PER_CONFIG + 1):
                        test_id += 1

                        # Generate graph configuration
                        graph_config_dict = generate_graph_config(
                            num_nodes=num_nodes,
                            weak_ratio=weak_ratio,
                            mandatory_ratio=mandatory_ratio,
                            seed=seed
                        )

                        all_nodes_list = (graph_config_dict['weak_nodes'] +
                                        graph_config_dict['power_nodes_mandatory'] +
                                        graph_config_dict['power_nodes_discretionary'])

                        # Generate capacities
                        cap_cfg = capacity_config.copy()
                        if 'random' in cap_cfg:
                            # Deep copy for random config
                            cap_cfg['random'] = cap_cfg['random'].copy()
                            if cap_cfg['random'].get('seed') is None:
                                cap_cfg['random']['seed'] = seed

                        capacities = generate_capacities(all_nodes_list, cap_cfg)

                        # Create shared graph with weight strategy
                        shared_graph = create_graph_with_strategy(
                            graph_config_dict['weak_nodes'],
                            graph_config_dict['power_nodes_mandatory'],
                            graph_config_dict['power_nodes_discretionary'],
                            capacities,
                            weight_strategy,
                            seed
                        )

                        # Prepare graph_config
                        graph_config = {
                            'weak_nodes': graph_config_dict['weak_nodes'],
                            'power_nodes_mandatory': graph_config_dict['power_nodes_mandatory'],
                            'power_nodes_discretionary': graph_config_dict['power_nodes_discretionary'],
                            'capacities': capacities
                        }

                        config_params = {
                            'num_nodes': num_nodes,
                            'weak_ratio': weak_ratio,
                            'mandatory_ratio': mandatory_ratio,
                            'capacity_config': cap_desc,
                            'weight_strategy': weight_strategy['name']
                        }

                        # Run test with shared graph
                        result = run_single_test_with_shared_graph(
                            shared_graph, graph_config, config_params, seed, test_id
                        )
                        all_results.append(result)

                        # Progress indicator every 100 tests
                        if test_id % 100 == 0:
                            elapsed = time.time() - start_time_bench
                            avg_per_test = elapsed / test_id
                            remaining = (total_tests - test_id) * avg_per_test
                            print(f"\n  Progress: {test_id}/{total_tests} ({test_id/total_tests*100:.1f}%) - "
                                  f"Est. remaining: {remaining/3600:.1f}h")

    end_time_bench = time.time()
    total_time = end_time_bench - start_time_bench

    # Save and aggregate results
    print(f"\n{'='*100}")
    print("SAVING RESULTS...")
    print(f"{'='*100}")

    # Raw results
    raw_file = f"{OUTPUT_DIR}/raw_results_{timestamp}.json"
    with open(raw_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  ✓ Raw results ({len(all_results)} tests): {raw_file}")

    # Calculate aggregated statistics
    print(f"\nCalculating aggregated statistics...")
    aggregated_stats = calculate_full_aggregation(all_results)

    agg_file = f"{OUTPUT_DIR}/aggregated_results_{timestamp}.json"
    with open(agg_file, 'w') as f:
        json.dump(aggregated_stats, f, indent=2)
    print(f"  ✓ Aggregated results: {agg_file}")

    # Generate report
    report_file = generate_comprehensive_report(all_results, aggregated_stats, timestamp, total_time)
    print(f"  ✓ Report: {report_file}")

    # Summary
    print(f"\n{'='*100}")
    print("BENCHMARK COMPLETE!")
    print(f"{'='*100}")
    print(f"Total tests: {len(all_results)}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Average per test: {total_time/len(all_results):.3f}s")

    total_greedy_matches = sum(1 for r in all_results if r.get('greedy', {}).get('matches_exhaustive', False))
    total_sa_matches = sum(1 for r in all_results if r.get('sa', {}).get('matches_exhaustive', False))

    print(f"\nOverall Match Rates:")
    print(f"  Greedy: {total_greedy_matches}/{len(all_results)} ({total_greedy_matches/len(all_results)*100:.1f}%)")
    print(f"  SA: {total_sa_matches}/{len(all_results)} ({total_sa_matches/len(all_results)*100:.1f}%)")

    return all_results, aggregated_stats


if __name__ == "__main__":
    print("\n🚀 Starting comprehensive factorial benchmark...\n")
    all_results, aggregated = run_benchmark()
    print("\n✅ Benchmark completed successfully!\n")
