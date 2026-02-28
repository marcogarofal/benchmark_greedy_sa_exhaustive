# benchmark_greedy_sa_exhaustive



python3 benchmark_multi_algorithm.py 7 --seeds 1 #exhaustive

## greedy and SimulatedAnnealing
python3 benchmark_multi_algorithm.py 8 --seeds 1
python3 benchmark_multi_algorithm.py 9 --seeds 1


python3 plot_multi_algorithm.py --all





## for debugging in "benchmark_multi_algorithm" modify:

    #debug_config = {'verbose': True, 'save_plots': False}
    debug_config = {'verbose': False, 'save_plots': False}


## plot:
python3 plot_multi_algorithm_v4.py --all --max-seeds 300
