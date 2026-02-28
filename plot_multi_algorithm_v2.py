"""
plot_multi_algorithm.py

Advanced plotting for multi-algorithm benchmark results

Usage:
    # Plot 1: Performance comparison (all ratios)
    python3 plot_multi_algorithm.py --performance
    
    # Plot 2+3: Quality metrics (50% discretionary only)
    python3 plot_multi_algorithm.py --quality-50pct
    
    # All plots
    python3 plot_multi_algorithm.py --all
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


OUTPUT_DIR = 'multi_algorithm_results'


def load_checkpoint(checkpoint_path):
    """Load checkpoint file"""
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)


def get_all_checkpoints():
    """Get all checkpoint files"""
    if not os.path.exists(OUTPUT_DIR):
        return []
    
    files = [f for f in os.listdir(OUTPUT_DIR) 
             if f.startswith('.checkpoint_') and f.endswith('.pkl')]
    
    checkpoints = []
    for f in files:
        path = os.path.join(OUTPUT_DIR, f)
        checkpoint = load_checkpoint(path)
        config_name = f.replace('.checkpoint_', '').replace('.pkl', '')
        checkpoints.append((config_name, checkpoint))
    
    return checkpoints


def plot_performance_all_ratios(save_path=None):
    """
    Plot 1: Performance comparison across all discretionary ratios
    3 subplots (10%, 30%, 50%), each with BAR CHART
    3 bars per num_nodes: exhaustive, greedy, SA
    """
    checkpoints = get_all_checkpoints()
    
    if not checkpoints:
        print("❌ No checkpoints found")
        return
    
    # Organize by ratio and algorithm
    data_by_ratio = {
        '10%': {'exhaustive': {}, 'greedy': {}, 'sa': {}},
        '30%': {'exhaustive': {}, 'greedy': {}, 'sa': {}},
        '50%': {'exhaustive': {}, 'greedy': {}, 'sa': {}}
    }
    
    for config_name, checkpoint in checkpoints:
        disc_ratio = checkpoint['configuration']['discretionary_ratio']
        algorithm = checkpoint['configuration']['algorithm']
        
        ratio_key = f"{int(disc_ratio*100)}%"
        
        if ratio_key not in data_by_ratio:
            continue
        
        # Extract time data for each node size
        for num_nodes in checkpoint['data'].keys():
            data = checkpoint['data'][num_nodes]
            successful = [t for t in data if t['metrics']['success']]
            
            if successful:
                times = [t['metrics']['execution_time'] for t in successful]
                data_by_ratio[ratio_key][algorithm][num_nodes] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'n': len(successful)
                }
    
    # Create plot with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Performance Comparison: Execution Time vs Node Size', 
                 fontsize=16, fontweight='bold')
    
    ratios = ['10%', '30%', '50%']
    colors = {'exhaustive': 'steelblue', 'greedy': 'coral', 'sa': 'mediumseagreen'}
    
    for idx, ratio in enumerate(ratios):
        ax = axes[idx]
        
        # Get common nodes for this ratio
        all_nodes = set()
        for algo_data in data_by_ratio[ratio].values():
            all_nodes.update(algo_data.keys())
        common_nodes = sorted(all_nodes)
        
        if not common_nodes:
            continue
        
        x = np.arange(len(common_nodes))
        width = 0.25
        
        # Plot bars for each algorithm
        for i, algorithm in enumerate(['exhaustive', 'greedy', 'sa']):
            means = [data_by_ratio[ratio][algorithm].get(n, {}).get('mean', 0) for n in common_nodes]
            stds = [data_by_ratio[ratio][algorithm].get(n, {}).get('std', 0) for n in common_nodes]
            
            offset = (i - 1) * width
            ax.bar(x + offset, means, width,
                  label=algorithm.upper(),
                  color=colors[algorithm],
                  yerr=stds,
                  capsize=4,
                  error_kw={'linewidth': 1.5})
        
        ax.set_xlabel('Number of Nodes', fontsize=12)
        ax.set_ylabel('Execution Time (s)', fontsize=12)
        ax.set_title(f'Discretionary {ratio}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(common_nodes)
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', which='both')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Performance plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_quality_50pct_time(save_path=None):
    """
    Plot 2: Execution time comparison for 50% discretionary
    Bar chart with error bars: num_nodes vs time for 3 algorithms
    """
    checkpoints = get_all_checkpoints()
    
    # Find 50% configs
    data_50pct = {'exhaustive': {}, 'greedy': {}, 'sa': {}}
    
    for config_name, checkpoint in checkpoints:
        disc_ratio = checkpoint['configuration']['discretionary_ratio']
        algorithm = checkpoint['configuration']['algorithm']
        
        if disc_ratio != 0.5:
            continue
        
        for num_nodes in checkpoint['data'].keys():
            data = checkpoint['data'][num_nodes]
            successful = [t for t in data if t['metrics']['success']]
            
            if successful:
                times = [t['metrics']['execution_time'] for t in successful]
                data_50pct[algorithm][num_nodes] = {
                    'mean': np.mean(times),
                    'std': np.std(times)
                }
    
    if not any(data_50pct.values()):
        print("❌ No data for 50% discretionary")
        return
    
    # Get common nodes
    all_nodes = set()
    for algo_data in data_50pct.values():
        all_nodes.update(algo_data.keys())
    common_nodes = sorted(all_nodes)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Quality Benchmark (50% Discretionary): Execution Time', 
                 fontsize=16, fontweight='bold')
    
    x = np.arange(len(common_nodes))
    width = 0.25
    
    colors = {'exhaustive': 'steelblue', 'greedy': 'coral', 'sa': 'mediumseagreen'}
    
    for i, algorithm in enumerate(['exhaustive', 'greedy', 'sa']):
        means = [data_50pct[algorithm].get(n, {}).get('mean', 0) for n in common_nodes]
        stds = [data_50pct[algorithm].get(n, {}).get('std', 0) for n in common_nodes]
        
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, 
               label=algorithm.upper(),
               color=colors[algorithm],
               yerr=stds,
               capsize=5,
               error_kw={'linewidth': 2})
    
    ax.set_xlabel('Number of Nodes', fontsize=14)
    ax.set_ylabel('Execution Time (s)', fontsize=14)
    ax.set_title('Mean Execution Time with Std Dev', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(common_nodes, fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3, axis='y', which='both')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Quality time plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_quality_50pct_metrics(save_path=None):
    """
    Plot 3: Quality metrics for 50% discretionary
    Shows how close greedy and SA are to exhaustive
    Multiple subplots: cost_ratio, match_rate, cost_diff, overlaps
    """
    checkpoints = get_all_checkpoints()
    
    # Find greedy and SA configs at 50%
    quality_data = {'greedy': {}, 'sa': {}}
    
    for config_name, checkpoint in checkpoints:
        disc_ratio = checkpoint['configuration']['discretionary_ratio']
        algorithm = checkpoint['configuration']['algorithm']
        has_quality = checkpoint['configuration'].get('has_quality', False)
        
        if disc_ratio != 0.5 or not has_quality:
            continue
        
        if algorithm not in ['greedy', 'sa']:
            continue
        
        # Extract quality statistics
        if 'statistics' in checkpoint:
            for num_nodes, stats in checkpoint['statistics'].items():
                if stats and 'quality' in stats:
                    quality_data[algorithm][num_nodes] = stats['quality']
    
    if not quality_data['greedy'] and not quality_data['sa']:
        print("❌ No quality data for 50% discretionary")
        print("   Run config 8-9 after config 7 (exhaustive)")
        return
    
    # Get common nodes
    all_nodes = set()
    for algo_data in quality_data.values():
        all_nodes.update(algo_data.keys())
    common_nodes = sorted(all_nodes)
    
    # Create plot with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Quality Metrics (50% Discretionary): Greedy & SA vs Exhaustive', 
                 fontsize=16, fontweight='bold')
    
    colors = {'greedy': 'coral', 'sa': 'mediumseagreen'}
    markers = {'greedy': 's', 'sa': '^'}
    
    # Plot 1: Cost Ratio (closer to 1.0 = better)
    ax = axes[0, 0]
    for algorithm in ['greedy', 'sa']:
        if not quality_data[algorithm]:
            continue
        
        nodes = sorted(quality_data[algorithm].keys())
        ratios = [quality_data[algorithm][n]['cost_ratio_mean'] for n in nodes]
        stds = [quality_data[algorithm][n]['cost_ratio_std'] for n in nodes]
        
        ax.errorbar(nodes, ratios, yerr=stds,
                   label=algorithm.upper(),
                   color=colors[algorithm],
                   marker=markers[algorithm],
                   markersize=10,
                   linewidth=2,
                   capsize=5)
    
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, 
              label='Perfect (same as exhaustive)')
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Cost Ratio (algo / exhaustive)', fontsize=12)
    ax.set_title('Cost Ratio: How Close to Optimal?', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Match Rate (% identical solutions)
    ax = axes[0, 1]
    for algorithm in ['greedy', 'sa']:
        if not quality_data[algorithm]:
            continue
        
        nodes = sorted(quality_data[algorithm].keys())
        match_rates = [quality_data[algorithm][n]['match_rate'] * 100 for n in nodes]
        
        ax.plot(nodes, match_rates,
               label=algorithm.upper(),
               color=colors[algorithm],
               marker=markers[algorithm],
               markersize=10,
               linewidth=2)
    
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Match Rate (%)', fontsize=12)
    ax.set_title('Match Rate: % Identical Solutions', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cost Difference (absolute)
    ax = axes[1, 0]
    for algorithm in ['greedy', 'sa']:
        if not quality_data[algorithm]:
            continue
        
        nodes = sorted(quality_data[algorithm].keys())
        diffs = [quality_data[algorithm][n]['cost_diff_mean'] for n in nodes]
        stds = [quality_data[algorithm][n]['cost_diff_std'] for n in nodes]
        
        ax.errorbar(nodes, diffs, yerr=stds,
                   label=algorithm.upper(),
                   color=colors[algorithm],
                   marker=markers[algorithm],
                   markersize=10,
                   linewidth=2,
                   capsize=5)
    
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Absolute Cost Difference', fontsize=12)
    ax.set_title('Cost Difference from Exhaustive', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Node & Edge Overlap
    ax = axes[1, 1]
    
    width = 0.35
    x = np.arange(len(common_nodes))
    
    for i, metric in enumerate(['node_overlap_mean', 'edge_overlap_mean']):
        metric_label = 'Node Overlap' if metric == 'node_overlap_mean' else 'Edge Overlap'
        
        for j, algorithm in enumerate(['greedy', 'sa']):
            if not quality_data[algorithm]:
                continue
            
            nodes = sorted(quality_data[algorithm].keys())
            values = [quality_data[algorithm][n][metric] * 100 for n in nodes]
            
            offset = (j - 0.5) * width + i * 2 * width
            
            ax.bar(x + offset, values, width * 0.8,
                  label=f'{algorithm.upper()} {metric_label}',
                  color=colors[algorithm],
                  alpha=0.7 if metric == 'edge_overlap_mean' else 1.0)
    
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Overlap (%)', fontsize=12)
    ax.set_title('Solution Overlap with Exhaustive', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(common_nodes)
    ax.set_ylim([0, 105])
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Quality metrics plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_quality_50pct_combined(save_path=None):
    """
    Combined quality plot for 50% discretionary
    Single plot with 3 bars per num_nodes showing all quality metrics
    """
    checkpoints = get_all_checkpoints()
    
    # Collect data for 50%
    quality_data = {'greedy': {}, 'sa': {}}
    
    for config_name, checkpoint in checkpoints:
        disc_ratio = checkpoint['configuration']['discretionary_ratio']
        algorithm = checkpoint['configuration']['algorithm']
        
        if disc_ratio != 0.5 or algorithm not in ['greedy', 'sa']:
            continue
        
        # Quality metrics
        if 'statistics' in checkpoint:
            for num_nodes, stats in checkpoint['statistics'].items():
                if stats and 'quality' in stats:
                    quality_data[algorithm][num_nodes] = stats['quality']
    
    if not quality_data['greedy'] and not quality_data['sa']:
        print("❌ No quality data for 50% discretionary")
        return
    
    # Get common nodes
    all_nodes = set()
    for algo_data in quality_data.values():
        all_nodes.update(algo_data.keys())
    common_nodes = sorted(all_nodes)
    
    # Create single plot with grouped bars
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle('Quality Metrics (50% Discretionary): Greedy & SA vs Exhaustive', 
                 fontsize=16, fontweight='bold')
    
    colors = {'greedy': 'coral', 'sa': 'mediumseagreen'}
    
    x = np.arange(len(common_nodes))
    width = 0.35
    
    # Prepare data: use cost_ratio as main metric (1.0 = perfect)
    greedy_ratios = []
    greedy_stds = []
    sa_ratios = []
    sa_stds = []
    
    for n in common_nodes:
        if n in quality_data['greedy']:
            greedy_ratios.append(quality_data['greedy'][n]['cost_ratio_mean'])
            greedy_stds.append(quality_data['greedy'][n]['cost_ratio_std'])
        else:
            greedy_ratios.append(0)
            greedy_stds.append(0)
        
        if n in quality_data['sa']:
            sa_ratios.append(quality_data['sa'][n]['cost_ratio_mean'])
            sa_stds.append(quality_data['sa'][n]['cost_ratio_std'])
        else:
            sa_ratios.append(0)
            sa_stds.append(0)
    
    # Plot bars
    bars1 = ax.bar(x - width/2, greedy_ratios, width,
                   label='GREEDY',
                   color=colors['greedy'],
                   yerr=greedy_stds,
                   capsize=5,
                   error_kw={'linewidth': 2, 'ecolor': 'darkred'})
    
    bars2 = ax.bar(x + width/2, sa_ratios, width,
                   label='SIMULATED ANNEALING',
                   color=colors['sa'],
                   yerr=sa_stds,
                   capsize=5,
                   error_kw={'linewidth': 2, 'ecolor': 'darkgreen'})
    
    # Perfect line (1.0 = same cost as exhaustive)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2.5, alpha=0.7,
              label='Perfect (1.0 = same as EXHAUSTIVE)')
    
    # Labels and formatting
    ax.set_xlabel('Number of Nodes', fontsize=14)
    ax.set_ylabel('Cost Ratio (algorithm / exhaustive)', fontsize=14)
    ax.set_title('Solution Quality: Cost Ratio vs Optimal (lower is better)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(common_nodes, fontsize=12)
    ax.legend(fontsize=13, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (gr, sa) in enumerate(zip(greedy_ratios, sa_ratios)):
        if gr > 0:
            ax.text(x[i] - width/2, gr + greedy_stds[i] + 0.01, 
                   f'{gr:.3f}', ha='center', va='bottom', fontsize=9, color='darkred')
        if sa > 0:
            ax.text(x[i] + width/2, sa + sa_stds[i] + 0.01,
                   f'{sa:.3f}', ha='center', va='bottom', fontsize=9, color='darkgreen')
    
    # Add text box with interpretation
    textstr = 'Cost Ratio Interpretation:\n1.00 = Perfect (same as exhaustive)\n1.05 = 5% worse\n1.10 = 10% worse'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Quality metrics plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_exhaustive_only(max_nodes=None, max_seeds=None, save_path=None):
    """
    Plot exhaustive algorithm for 10%, 30%, 50% discretionary ratios
    Bar chart: num_nodes (x-axis) vs execution time (y-axis)
    3 bars per num_nodes (one for each discretionary ratio)
    
    Args:
        max_nodes: Maximum number of nodes to include (e.g., 9 to stop at 9 nodes)
        max_seeds: Maximum number of seeds to include (e.g., 400 to use first 400 seeds)
        save_path: Path to save the plot
    """
    checkpoints = get_all_checkpoints()
    
    if not checkpoints:
        print("❌ No checkpoints found")
        return
    
    # Organize data by discretionary ratio (only exhaustive)
    data_by_ratio = {
        '10%': {},
        '30%': {},
        '50%': {}
    }
    
    for config_name, checkpoint in checkpoints:
        disc_ratio = checkpoint['configuration']['discretionary_ratio']
        algorithm = checkpoint['configuration']['algorithm']
        
        # Only process exhaustive algorithm
        if algorithm != 'exhaustive':
            continue
        
        ratio_key = f"{int(disc_ratio*100)}%"
        
        if ratio_key not in data_by_ratio:
            continue
        
        # Extract time data for each node size
        for num_nodes in checkpoint['data'].keys():
            # Apply max_nodes filter
            if max_nodes is not None and num_nodes > max_nodes:
                continue
            
            data = checkpoint['data'][num_nodes]
            
            # Apply max_seeds filter
            if max_seeds is not None:
                data = data[:max_seeds]
            
            # Only successful runs
            successful = [t for t in data if t['metrics']['success']]
            
            if successful:
                times = [t['metrics']['execution_time'] for t in successful]
                data_by_ratio[ratio_key][num_nodes] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'n': len(successful)
                }
    
    # Check if we have data
    has_data = any(data_by_ratio[r] for r in data_by_ratio)
    if not has_data:
        print("❌ No exhaustive data found")
        return
    
    # Get common nodes across all ratios
    all_nodes = set()
    for ratio_data in data_by_ratio.values():
        all_nodes.update(ratio_data.keys())
    common_nodes = sorted(all_nodes)
    
    if not common_nodes:
        print("❌ No common nodes found")
        return
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Build title with filters info
    title = 'Exhaustive Algorithm: Execution Time Comparison'
    filter_info = []
    if max_nodes is not None:
        filter_info.append(f'max_nodes={max_nodes}')
    if max_seeds is not None:
        filter_info.append(f'max_seeds={max_seeds}')
    if filter_info:
        title += f'\n({", ".join(filter_info)})'
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Colors for each discretionary ratio
    colors = {
        '10%': 'skyblue',
        '30%': 'steelblue', 
        '50%': 'navy'
    }
    
    x = np.arange(len(common_nodes))
    width = 0.25
    
    # Plot bars for each discretionary ratio
    ratios = ['10%', '30%', '50%']
    for i, ratio in enumerate(ratios):
        means = []
        stds = []
        
        for n in common_nodes:
            if n in data_by_ratio[ratio]:
                means.append(data_by_ratio[ratio][n]['mean'])
                stds.append(data_by_ratio[ratio][n]['std'])
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width,
                     label=f'Discretionary {ratio}',
                     color=colors[ratio],
                     yerr=stds,
                     capsize=5,
                     error_kw={'linewidth': 2})
        
        # Add value labels on top of bars
        for j, (mean, std) in enumerate(zip(means, stds)):
            if mean > 0:
                # Format time value
                if mean < 0.01:
                    label = f'{mean*1000:.2f}ms'
                elif mean < 1:
                    label = f'{mean:.3f}s'
                else:
                    label = f'{mean:.2f}s'
                
                ax.text(x[j] + offset, mean + std + max(means)*0.02,
                       label, ha='center', va='bottom', 
                       fontsize=9, rotation=0)
    
    # Labels and formatting
    ax.set_xlabel('Number of Nodes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(common_nodes, fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', which='both', linestyle='--')
    
    # Add info box
    info_text = 'EXHAUSTIVE Algorithm Only\n'
    info_text += f'Nodes plotted: {min(common_nodes)} to {max(common_nodes)}\n'
    
    # Count seeds used
    sample_ratio = list(data_by_ratio.keys())[0]
    if common_nodes and common_nodes[0] in data_by_ratio[sample_ratio]:
        n_samples = data_by_ratio[sample_ratio][common_nodes[0]]['n']
        info_text += f'Seeds per node: {n_samples}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Exhaustive plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary statistics
    print("\n📈 Summary Statistics (Exhaustive):")
    print("=" * 60)
    for ratio in ratios:
        if not data_by_ratio[ratio]:
            continue
        print(f"\n{ratio} Discretionary:")
        for n in sorted(data_by_ratio[ratio].keys()):
            mean = data_by_ratio[ratio][n]['mean']
            std = data_by_ratio[ratio][n]['std']
            count = data_by_ratio[ratio][n]['n']
            print(f"  {n} nodes: {mean:.4f}s ± {std:.4f}s (n={count})")


def main():
    """Main plotting function"""
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ Directory '{OUTPUT_DIR}' not found")
        return
    
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python3 plot_multi_algorithm.py --performance")
        print("  python3 plot_multi_algorithm.py --quality-50pct")
        print("  python3 plot_multi_algorithm.py --exhaustive [--max-nodes N] [--max-seeds N]")
        print("  python3 plot_multi_algorithm.py --all")
        print("\nExamples:")
        print("  python3 plot_multi_algorithm.py --exhaustive --max-nodes 9")
        print("  python3 plot_multi_algorithm.py --exhaustive --max-seeds 400")
        print("  python3 plot_multi_algorithm.py --exhaustive --max-nodes 9 --max-seeds 400")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if '--performance' in sys.argv or '--all' in sys.argv:
        print("📊 Generating performance comparison plot...")
        save_path = os.path.join(OUTPUT_DIR, f'performance_all_ratios_{timestamp}.png')
        plot_performance_all_ratios(save_path)
    
    if '--quality-50pct' in sys.argv or '--all' in sys.argv:
        print("📊 Generating quality plots for 50% discretionary...")
        save_path = os.path.join(OUTPUT_DIR, f'quality_50pct_combined_{timestamp}.png')
        plot_quality_50pct_combined(save_path)
    
    if '--exhaustive' in sys.argv or '--all' in sys.argv:
        print("📊 Generating exhaustive-only comparison plot...")
        
        # Parse max_nodes and max_seeds arguments
        max_nodes = None
        max_seeds = None
        
        try:
            if '--max-nodes' in sys.argv:
                idx = sys.argv.index('--max-nodes')
                max_nodes = int(sys.argv[idx + 1])
                print(f"   Max nodes: {max_nodes}")
            
            if '--max-seeds' in sys.argv:
                idx = sys.argv.index('--max-seeds')
                max_seeds = int(sys.argv[idx + 1])
                print(f"   Max seeds: {max_seeds}")
        except (IndexError, ValueError) as e:
            print(f"⚠️  Error parsing arguments: {e}")
            print("   Using default values (all nodes and seeds)")
        
        # Generate filename with filters
        filename = f'exhaustive_comparison_{timestamp}'
        if max_nodes:
            filename += f'_maxnodes{max_nodes}'
        if max_seeds:
            filename += f'_maxseeds{max_seeds}'
        filename += '.png'
        
        save_path = os.path.join(OUTPUT_DIR, filename)
        plot_exhaustive_only(max_nodes=max_nodes, max_seeds=max_seeds, save_path=save_path)
    
    print("\n✅ All plots generated!")





if __name__ == "__main__":
    main()
