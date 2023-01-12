import numpy as np
import json
import sys
import copy

import bandit_non_stationarity.epsilon_greedy, bandit_non_stationarity.ucb, \
    factored_bandit_distributional.epsilon_greedy, factored_bandit_distributional.ucb

import test_statistics
import time

from randomization_tests import mc_construct_rand_p_value, shared_mc_construct_rand_p_value

start = time.time()

num_samples = int(sys.argv[7])
epsilon = float(sys.argv[6])
style = str(sys.argv[5])
algo_name = str(sys.argv[4])
j = int(sys.argv[3])
T = int(sys.argv[2])
trials = int(sys.argv[1])

alpha = 0.05
styles = ['u', 'i_X', 'i', 'r', 'c', 'uu_X', 'ui_X', 'rui_X', 'iu_X', 'ii_X', 'comb']

np.random.seed([T, j, styles.index(style), int(100*epsilon), \
    num_samples, trials])

null = True # dummy setting of null; doesn't matter

lengths = []
coverage = []
# iterate through trials
for job_ind in range(trials):
    counts = np.zeros(11)
    if job_ind % 10 == 0:
        print(job_ind)

    test_stat = test_statistics.bandit_non_stationarity 
    if algo_name == 'epsilon_greedy':
        algorithm = bandit_non_stationarity.epsilon_greedy.EpsilonGreedy(T, epsilon, null, True if style=='c' else False)
    if algo_name == 'ucb':
        algorithm = bandit_non_stationarity.ucb.UCB(T, null, True if style=='c' else False)

    # get the true dataset
    true_data = algorithm.get_dataset()

    last_action = true_data[-1][0]
    b_vals = np.linspace(-5,5,11)
    
    # #### 
    # true_data[3] = (true_data[3][0], b_vals[0])
    # copy_data = copy.deepcopy(true_data)
    # copy_data[3] = (copy_data[3][0], b_vals[3])

    # print("----")
    # print(algorithm.get_data_weight(true_data), algorithm.get_data_weight(copy_data))
    # print("***")
    # print(algorithm.get_shared_data_weight(true_data, b_vals)[0], algorithm.get_shared_data_weight(true_data, b_vals)[3])
    # print("----")
    # continue
    # #####

    p_pluses, p_minuses  = shared_mc_construct_rand_p_value(algorithm, true_data, test_stat, style, b_vals, num_samples)

    for indexer in range(len(b_vals)):
        p_plus = p_pluses[indexer]
        p_minus = p_minuses[indexer]

        # calculate acceptance or not
        if p_plus > alpha:
            counts[indexer] = 1
        else:
            counts[indexer] = 0
    
    # calculate length on the interval by using rounding of Chen, Chun, 
    # and Barber (2017) (to nearest integer)
    length = 0
    for indexer in range(11):
        if indexer == 0 or indexer == 10:
            if counts[indexer] == 1:
                length += 0.5
        else:
            if counts[indexer] == 1:
                length += 1.

    lengths.append(length)

    y_last = true_data[-1][1]
    rounded_y = round(y_last)

    if rounded_y in b_vals:
        coverage.append(counts[np.where(b_vals==rounded_y)[0][0]])
    else:
        coverage.append(0)
print(lengths)
print(np.mean(lengths), np.std(lengths)/np.sqrt(len(lengths)))
print(np.mean(coverage), np.std(coverage)/np.sqrt(len(coverage)))


# save file
length_file_name = f'length_shared/{algo_name}_{T}_{num_samples}_{j}_{style}_{epsilon}'
coverage_file_name = f'coverage_shared/{algo_name}_{T}_{num_samples}_{j}_{style}_{epsilon}'

end = time.time()

elapsed_time = end-start


with open(length_file_name, 'w+') as f:
        json.dump(lengths, f)


# save time in the length file
with open(length_file_name + '_time', 'w+') as f:
    json.dump(elapsed_time, f)

with open(coverage_file_name, 'w+') as f:
        json.dump(coverage, f)

