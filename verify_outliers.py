import pickle
import numpy as np

# Config 2: 10% greedy
with open('multi_algorithm_results/.checkpoint_10pct_greedy.pkl', 'rb') as f:
    cp = pickle.load(f)

print("="*80)
print("CONFIG 2: 10% GREEDY - NODE 8 ANALYSIS")
print("="*80)

times = [t['metrics']['execution_time'] for t in cp['data'][8] if t['metrics']['success']]

print(f"Number of successful tests: {len(times)}/200")
print(f"\nTime statistics:")
print(f"  Mean: {np.mean(times):.6f}s")
print(f"  Std:  {np.std(times):.6f}s")
print(f"  Min:  {np.min(times):.6f}s")
print(f"  Max:  {np.max(times):.6f}s")
print(f"  Median: {np.median(times):.6f}s")

# Trova outliers
q1 = np.percentile(times, 25)
q3 = np.percentile(times, 75)
iqr = q3 - q1
outliers = [t for t in times if t < q1 - 1.5*iqr or t > q3 + 1.5*iqr]

print(f"\nOutliers: {len(outliers)}")
if outliers:
    print(f"  Outlier times: {sorted(outliers)}")
    print(f"  Max outlier is {max(outliers)/np.median(times):.1f}x the median")

# Mostra distribuzione
print(f"\nDistribution:")
print(f"  < 0.001s: {sum(1 for t in times if t < 0.001)}")
print(f"  0.001-0.01s: {sum(1 for t in times if 0.001 <= t < 0.01)}")
print(f"  0.01-0.1s: {sum(1 for t in times if 0.01 <= t < 0.1)}")
print(f"  > 0.1s: {sum(1 for t in times if t >= 0.1)}")














# Node 5 dovrebbe essere stabile
times_5 = [t['metrics']['execution_time'] for t in cp['data'][5] if t['metrics']['success']]

print("\nNODE 5:")
print(f"  Mean: {np.mean(times_5):.6f}s")
print(f"  Std:  {np.std(times_5):.6f}s")
print(f"  Max/Median ratio: {np.max(times_5)/np.median(times_5):.1f}x")

# Config 5: 30% greedy node 6
with open('multi_algorithm_results/.checkpoint_30pct_greedy.pkl', 'rb') as f:
    cp30 = pickle.load(f)

times_30_6 = [t['metrics']['execution_time'] for t in cp30['data'][6] if t['metrics']['success']]

print("\nCONFIG 5 (30%) - NODE 6:")
print(f"  Mean: {np.mean(times_30_6):.6f}s")
print(f"  Std:  {np.std(times_30_6):.6f}s")
print(f"  Max/Median ratio: {np.max(times_30_6)/np.median(times_30_6):.1f}x")
