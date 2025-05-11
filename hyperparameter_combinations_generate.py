from itertools import product
import itertools

from classes.keep_track_class import *
from classes.saving_class import *
# Saving all json files into one CSV file:
print('stating ...')
saving_module = Saving()
csv_file_path = saving_module.convert_json_to_csv('results')

hyperparameters = ["optimizer_type","problem_type","learning_rate","num_epochs","batch_size","epsilon","delta","clip_bound","seed"]
tracker = HyperparameterTracker(csv_file_path,hyperparameters,'/scratch/project_2003275/Andrew_temp/Benchmarking-Differential-Privacy-Optimizers/results/hyperparameters_tracking.csv')
tracker_file_name = tracker.extract_hyperparameters()
optimizer_types = ['adambc']
if optimizer_types[0] =='adambc':
    learning_rates = [0.000001,0.00001,0.0001,0.001] # adam
else:
    learning_rates = [0.001, 0.01 , 0.1,1  ] 

batch_sizes = [ 256,512,1024,2048]
epsilons = [1, 5, 10]
seeds = [2240989660,1443617675,13242043,1023176085,2008379934]
num_epochs = [250,500]

deltas = [1e-5]
clip_bounds = [1]
problem_types = [1,2] 
combinations = [
        comb for comb in itertools.product(optimizer_types, learning_rates, batch_sizes, epsilons, seeds, num_epochs, deltas, clip_bounds, problem_types)
        if not tracker.has_run(optimizer_type=comb[0], learning_rate=comb[1], batch_size=comb[2], epsilon=comb[3], seed=comb[4], num_epochs=comb[5], delta=comb[6], clip_bound=comb[7], problem_type=comb[8])
    ]



# Save the combinations to a file
with open('combinations.txt', 'w') as f:
    for comb in combinations:
        f.write(','.join(map(str, comb)) + '\n')

print("Combinations have been saved to combinations.txt")


