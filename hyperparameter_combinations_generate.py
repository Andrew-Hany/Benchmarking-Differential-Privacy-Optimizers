from itertools import product
import itertools

from classes.keep_track_class import *

hyperparameters = ["optimizer_type", "problem_type", "learning_rate", "num_epochs", "batch_size", "epsilon", "delta", "clip_bound", "seed"]
tracker = HyperparameterTracker('/scratch/project_2003275/Andrew_temp/Benchmarking-Differential-Privacy-Optimizers/hyperparameter_combinations_generate.py',hyperparameters,'/scratch/project_2003275/Andrew_temp/Benchmarking-Differential-Privacy-Optimizers/results/hyperparameters_tracking.csv')


optimizer_types = ['sgd']
# learning_rates = np.logspace(np.log10(0.1), np.log10(0.7), num=4) #sgd
learning_rates = [0.001, 0.01 , 0.1  , 1.   ]
batch_sizes = [128, 256]
epsilons = [1, 5, 10]
seeds = [51, 92, 14, 71]

num_epochs = [100,200]
deltas = [1e-5]
clip_bounds = [1]
problem_types = [2]
combinations = [
        comb for comb in itertools.product(optimizer_types, learning_rates, batch_sizes, epsilons, seeds, num_epochs, deltas, clip_bounds, problem_types)
        if not tracker.has_run(optimizer_type=comb[0], learning_rate=comb[1], batch_size=comb[2], epsilon=comb[3], seed=comb[4], num_epochs=comb[5], delta=comb[6], clip_bound=comb[7], problem_type=comb[8])
    ]



# Save the combinations to a file
with open('combinations.txt', 'w') as f:
    for comb in combinations:
        f.write(','.join(map(str, comb)) + '\n')

print("Combinations have been saved to combinations.txt")


