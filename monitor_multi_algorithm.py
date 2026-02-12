"""
monitor_multi_algorithm.py

Monitor progress for multi-algorithm benchmark

Usage:
    python3 monitor_multi_algorithm.py
    watch -n 5 python3 monitor_multi_algorithm.py
"""

import os
import pickle
import sys
from datetime import datetime


OUTPUT_DIR = 'multi_algorithm_results'


def load_checkpoint(checkpoint_path):
    """Load checkpoint file"""
    try:
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    except:
        return None


def format_time(seconds):
    """Format seconds to human readable"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.2f}h"


def monitor_all():
    """Monitor all configurations"""
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ Directory '{OUTPUT_DIR}' not found")
        return
    
    checkpoint_files = [
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith('.checkpoint_') and f.endswith('.pkl')
    ]
    
    if not checkpoint_files:
        print(f"❌ No checkpoints found in {OUTPUT_DIR}/")
        return
    
    # Check locks
    lock_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('.lock_')]
    running_configs = set(f.replace('.lock_', '') for f in lock_files)
    
    print(f"\n{'='*100}")
    print(f"MULTI-ALGORITHM BENCHMARK PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")
    
    # Group by discretionary ratio
    by_ratio = {'10%': [], '30%': [], '50%': []}
    
    for checkpoint_file in sorted(checkpoint_files):
        checkpoint_path = os.path.join(OUTPUT_DIR, checkpoint_file)
        checkpoint = load_checkpoint(checkpoint_path)
        
        if checkpoint is None:
            continue
        
        config_name = checkpoint_file.replace('.checkpoint_', '').replace('.pkl', '')
        is_running = config_name in running_configs
        
        disc_ratio = checkpoint['configuration']['discretionary_ratio']
        algorithm = checkpoint['configuration']['algorithm']
        has_quality = checkpoint['configuration'].get('has_quality', False)
        
        ratio_key = f"{int(disc_ratio*100)}%"
        
        info = {
            'name': config_name,
            'algorithm': algorithm,
            'running': is_running,
            'seeds': checkpoint.get('max_seed', 0),
            'nodes': sorted(checkpoint['data'].keys()),
            'time': checkpoint.get('total_execution_time', 0),
            'has_quality': has_quality
        }
        
        # Get quality info if available
        if has_quality and 'statistics' in checkpoint:
            stats = checkpoint['statistics']
            if stats:
                sample_node = next(iter(stats.keys()))
                if stats[sample_node] and 'quality' in stats[sample_node]:
                    q = stats[sample_node]['quality']
                    info['match_rate'] = q['match_rate']
                    info['cost_ratio'] = q['cost_ratio_mean']
        
        by_ratio[ratio_key].append(info)
    
    # Display by ratio
    for ratio in ['10%', '30%', '50%']:
        if not by_ratio[ratio]:
            continue
        
        print(f"{'─'*100}")
        print(f"DISCRETIONARY {ratio}")
        print(f"{'─'*100}")
        
        for info in by_ratio[ratio]:
            status = "🔄 RUNNING" if info['running'] else "✓ STOPPED"
            quality_info = ""
            
            if info.get('has_quality'):
                if 'match_rate' in info:
                    quality_info = f", match={info['match_rate']*100:.0f}%, ratio={info['cost_ratio']:.3f}"
                else:
                    quality_info = " [quality pending]"
            
            print(f"{status} {info['algorithm'].upper():<12} | "
                  f"seeds={info['seeds']:<4} | "
                  f"nodes={info['nodes']} | "
                  f"time={format_time(info['time'])}{quality_info}")
        
        print()
    
    print(f"{'='*100}\n")


def main():
    monitor_all()


if __name__ == "__main__":
    main()
