import pickle
from graph_generator import generate_graph_config

# Carica checkpoint
with open('multi_algorithm_results/.checkpoint_50pct_exhaustive.pkl', 'rb') as f:
    cp_ex = pickle.load(f)

with open('multi_algorithm_results/.checkpoint_50pct_greedy.pkl', 'rb') as f:
    cp_gr = pickle.load(f)

with open('multi_algorithm_results/.checkpoint_50pct_sa.pkl', 'rb') as f:
    cp_sa = pickle.load(f)

print("="*100)
print("COMPLETE SOLUTION COMPARISON - NODE 5, SEED 1")
print("="*100)

num_nodes = 8
seed = 1

# Ricalcola configurazione ESATTA come fa run_single_test()
disc_ratio = 0.5
weak_ratio = 0.4
mandatory_ratio = 0.3

num_discretionary = int(num_nodes * disc_ratio)
remaining_nodes = num_nodes - num_discretionary
total_base_ratio = weak_ratio + mandatory_ratio
num_weak_base = int(remaining_nodes * (weak_ratio / total_base_ratio))
num_mandatory_base = remaining_nodes - num_weak_base

# Genera la configurazione esatta
graph_config_dict = generate_graph_config(
    num_weak=num_weak_base,
    num_mandatory=num_mandatory_base,
    num_discretionary=num_discretionary,
    seed=seed
)

weak_nodes = graph_config_dict['weak_nodes']
mandatory_nodes = graph_config_dict['power_nodes_mandatory']
discretionary_nodes = graph_config_dict['power_nodes_discretionary']

print(f"\nCONFIGURATION (num_nodes={num_nodes}, disc_ratio={disc_ratio}):")
print(f"  Discretionary: {num_discretionary} → {discretionary_nodes}")
print(f"  Remaining: {remaining_nodes}")
print(f"  Weak: {num_weak_base} → {weak_nodes}")
print(f"  Mandatory: {num_mandatory_base} → {mandatory_nodes}")
print(f"  Total: {len(weak_nodes) + len(mandatory_nodes) + len(discretionary_nodes)}")

# Estrai metriche
ex_metrics = cp_ex['data'][num_nodes][0]['metrics']
gr_metrics = cp_gr['data'][num_nodes][0]['metrics']
sa_metrics = cp_sa['data'][num_nodes][0]['metrics']

# EXHAUSTIVE
print("\n" + "="*100)
print("EXHAUSTIVE SOLUTION")
print("="*100)
print(f"Nodes ({ex_metrics['num_nodes']}): {sorted(ex_metrics['solution_nodes'])}")
print(f"Edges ({ex_metrics['num_edges']}): {sorted(set(tuple(sorted(e)) for e in ex_metrics['solution_edges']))}")
print(f"\nNode analysis:")
ex_nodes = set(ex_metrics['solution_nodes'])
ex_weak_used = ex_nodes & set(weak_nodes)
ex_mand_used = ex_nodes & set(mandatory_nodes)
ex_disc_used = ex_nodes & set(discretionary_nodes)
print(f"  Weak used: {sorted(ex_weak_used)} ({len(ex_weak_used)}/{len(weak_nodes)})")
print(f"  Mandatory used: {sorted(ex_mand_used)} ({len(ex_mand_used)}/{len(mandatory_nodes)})")
print(f"  Discretionary used: {sorted(ex_disc_used)} ({len(ex_disc_used)}/{len(discretionary_nodes)})")
print(f"\nCosts:")
print(f"  Edge cost:   {ex_metrics['edge_cost']:.6f}")
print(f"  Degree cost: {ex_metrics['degree_cost']:.6f}")
print(f"  TOTAL:       {ex_metrics['total_cost']:.6f}")

# GREEDY
print("\n" + "="*100)
print("GREEDY SOLUTION")
print("="*100)
print(f"Nodes ({gr_metrics['num_nodes']}): {sorted(gr_metrics['solution_nodes'])}")
print(f"Edges ({gr_metrics['num_edges']}): {sorted(set(tuple(sorted(e)) for e in gr_metrics['solution_edges']))}")
print(f"\nNode analysis:")
gr_nodes = set(gr_metrics['solution_nodes'])
gr_weak_used = gr_nodes & set(weak_nodes)
gr_mand_used = gr_nodes & set(mandatory_nodes)
gr_disc_used = gr_nodes & set(discretionary_nodes)
print(f"  Weak used: {sorted(gr_weak_used)} ({len(gr_weak_used)}/{len(weak_nodes)})")
print(f"  Mandatory used: {sorted(gr_mand_used)} ({len(gr_mand_used)}/{len(mandatory_nodes)})")
print(f"  Discretionary used: {sorted(gr_disc_used)} ({len(gr_disc_used)}/{len(discretionary_nodes)})")
print(f"\nCosts:")
print(f"  Edge cost:   {gr_metrics['edge_cost']:.6f}")
print(f"  Degree cost: {gr_metrics['degree_cost']:.6f}")
print(f"  TOTAL:       {gr_metrics['total_cost']:.6f}")

# SA
print("\n" + "="*100)
print("SIMULATED ANNEALING SOLUTION")
print("="*100)
print(f"Nodes ({sa_metrics['num_nodes']}): {sorted(sa_metrics['solution_nodes'])}")
print(f"Edges ({sa_metrics['num_edges']}): {sorted(set(tuple(sorted(e)) for e in sa_metrics['solution_edges']))}")
print(f"\nNode analysis:")
sa_nodes = set(sa_metrics['solution_nodes'])
sa_weak_used = sa_nodes & set(weak_nodes)
sa_mand_used = sa_nodes & set(mandatory_nodes)
sa_disc_used = sa_nodes & set(discretionary_nodes)
print(f"  Weak used: {sorted(sa_weak_used)} ({len(sa_weak_used)}/{len(weak_nodes)})")
print(f"  Mandatory used: {sorted(sa_mand_used)} ({len(sa_mand_used)}/{len(mandatory_nodes)})")
print(f"  Discretionary used: {sorted(sa_disc_used)} ({len(sa_disc_used)}/{len(discretionary_nodes)})")
print(f"\nCosts:")
print(f"  Edge cost:   {sa_metrics['edge_cost']:.6f}")
print(f"  Degree cost: {sa_metrics['degree_cost']:.6f}")
print(f"  TOTAL:       {sa_metrics['total_cost']:.6f}")

# COMPARISON TABLE
print("\n" + "="*100)
print("COMPARISON TABLE")
print("="*100)
print(f"{'Algorithm':<15} {'Nodes':<8} {'Edges':<8} {'Disc':<8} {'Total Cost':<15} {'Ratio':<12} {'Status':<15}")
print("-"*100)

ex_cost = ex_metrics['total_cost']
gr_cost = gr_metrics['total_cost']
sa_cost = sa_metrics['total_cost']

gr_ratio = gr_cost / ex_cost if ex_cost > 0 and ex_cost != float('inf') else float('inf')
sa_ratio = sa_cost / ex_cost if ex_cost > 0 and ex_cost != float('inf') else float('inf')

ex_status = "✅ Valid" if ex_cost != float('inf') else "❌ Invalid"
gr_status = "✅ Valid" if gr_cost != float('inf') else "❌ Invalid"
sa_status = "✅ Valid" if sa_cost != float('inf') else "❌ Invalid"

if gr_ratio < 1.0 and gr_cost != float('inf'):
    gr_status = "🚨 BETTER THAN EX"
if sa_ratio < 1.0 and sa_cost != float('inf'):
    sa_status = "🚨 BETTER THAN EX"

print(f"{'Exhaustive':<15} {ex_metrics['num_nodes']:<8} {ex_metrics['num_edges']:<8} {len(ex_disc_used):<8} {ex_cost:<15.6f} {'1.000':<12} {ex_status:<15}")
print(f"{'Greedy':<15} {gr_metrics['num_nodes']:<8} {gr_metrics['num_edges']:<8} {len(gr_disc_used):<8} {gr_cost:<15.6f} {f'{gr_ratio:.3f}':<12} {gr_status:<15}")
print(f"{'SA':<15} {sa_metrics['num_nodes']:<8} {sa_metrics['num_edges']:<8} {len(sa_disc_used):<8} {sa_cost:<15.6f} {f'{sa_ratio:.3f}':<12} {sa_status:<15}")

# TOPOLOGY VALIDATION
print("\n" + "="*100)
print("TOPOLOGY VALIDATION")
print("="*100)

for algo_name, metrics in [('Exhaustive', ex_metrics), ('Greedy', gr_metrics), ('SA', sa_metrics)]:
    print(f"\n{algo_name}:")
    
    if metrics['num_nodes'] == 0 or metrics['total_cost'] == float('inf'):
        print("  ❌ NO VALID SOLUTION")
        continue
    
    nodes = set(metrics['solution_nodes'])
    edges = [tuple(sorted(e)) for e in metrics['solution_edges']]
    
    # Check all weak and mandatory present
    missing_weak = set(weak_nodes) - nodes
    missing_mandatory = set(mandatory_nodes) - nodes
    
    if missing_weak:
        print(f"  ❌ Missing weak nodes: {sorted(missing_weak)}")
    if missing_mandatory:
        print(f"  ❌ Missing mandatory nodes: {sorted(missing_mandatory)}")
    
    if not missing_weak and not missing_mandatory:
        print(f"  ✅ All weak and mandatory present")
    
    # Build connectivity graph
    node_connections = {n: [] for n in nodes}
    for u, v in edges:
        node_connections[u].append(v)
        node_connections[v].append(u)
    
    # Check weak nodes (max 1 connection)
    weak_violations = []
    for w in weak_nodes:
        if w in node_connections and len(node_connections[w]) > 1:
            weak_violations.append((w, len(node_connections[w])))
    
    if weak_violations:
        print(f"  ❌ Weak nodes as hubs: {weak_violations}")
    else:
        print(f"  ✅ No weak nodes are hubs")
    
    # Check discretionary (must connect to weak or mandatory)
    disc_violations = []
    for d in discretionary_nodes:
        if d in node_connections:
            neighbors = node_connections[d]
            connects_to_weak_or_mand = any(
                n in weak_nodes or n in mandatory_nodes 
                for n in neighbors
            )
            if not connects_to_weak_or_mand:
                disc_violations.append((d, neighbors))
    
    if disc_violations:
        for d, neighbors in disc_violations:
            print(f"  🚨 Discretionary {d} only connects to other discretionary: {neighbors}")
    else:
        if any(d in nodes for d in discretionary_nodes):
            print(f"  ✅ All discretionary nodes are useful (connect to weak or mandatory)")

print("\n" + "="*100)
