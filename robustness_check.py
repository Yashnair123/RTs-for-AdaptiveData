import numpy as np
import sys
import json

import test_statistics
from bandit_non_stationarity import epsilon_greedy

from randomization_tests import mc_construct_rand_p_value_with_weights



num_samples = 100
epsilons = np.linspace(0.1,0.9,9)
null = True
mc = True
T = 100
trials = 1000

variants = []


job = int(sys.argv[1])
np.random.seed(job)

for style in ['u', 'i', 'r']:
    for analysis_eps in epsilons:
        for true_eps in epsilons:
            for block in range(4):
                variants.append([style,analysis_eps,true_eps,block])

style,analysis_eps,true_eps,block=variants[job]


alpha = 0.05
styles = ['u', 'i', 'r']


test_stat = test_statistics.bandit_non_stationarity

results = dict()
counts = []
inflations = []
for _ in range(block*250, (block+1)*250):
    true_algorithm = epsilon_greedy.EpsilonGreedy(T, true_eps, null, False)
    true_data = true_algorithm.get_dataset()
    analysis_algorithm = epsilon_greedy.EpsilonGreedy(T, analysis_eps, null, False)
    p_plus, p_minus, eff_sample_ratio, est_weights, true_weights = mc_construct_rand_p_value_with_weights(analysis_algorithm, true_algorithm, true_data, test_stat, style, num_samples)
    
    # calculate the randomized p-value
    if p_minus > alpha:
        counts.append(0)
    elif p_plus > alpha:
        counts.append(1-int(np.random.uniform() < (p_plus - alpha)/(p_plus - p_minus)))
    else:
        counts.append(1)

    inflation = 0.5*np.sum(np.absolute(true_weights-est_weights))
    #print(inflation)
    inflations.append(inflation)

#print(f"Results for {analysis_eps}, {style}: {np.mean(counts)}, {np.std(counts)/np.sqrt(len(counts))}, {alpha+np.mean(inflations)}")
# results = [np.mean(counts), np.std(counts)/np.sqrt(len(counts)), alpha+np.mean(inflations), np.std(inflations)/np.sqrt(len(inflations))]
# save file
file_name=f'robustness_results/counts_egreedy_{style}_{analysis_eps}_{true_eps}_{block}'
with open(file_name, 'w+') as f:
    json.dump(counts, f)

file_name=f'robustness_results/inflations_egreedy_{style}_{analysis_eps}_{true_eps}_{block}'
with open(file_name, 'w+') as f:
    json.dump(inflations, f)