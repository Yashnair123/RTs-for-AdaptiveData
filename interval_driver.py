import numpy as np
import json
import sys
import copy

import bandit_non_stationarity.epsilon_greedy, bandit_non_stationarity.ucb, \
    factored_bandit_distributional.epsilon_greedy, factored_bandit_distributional.ucb

import test_statistics
import time

from randomization_tests import conf_interval_mcmc_construct_rand_p_value, conf_interval_mc_construct_rand_p_value,\
    mcmc_construct_rand_p_value, mc_construct_rand_p_value


start = time.time()

type = str(sys.argv[9])
num_samples = int(sys.argv[8])
epsilon = float(sys.argv[7])
mc = eval(str(sys.argv[6]))
style = str(sys.argv[5])
algo_name = str(sys.argv[4])
j = int(sys.argv[3])
T = int(sys.argv[2])
trials = int(sys.argv[1])

alpha = 0.05
styles = ['u', 'i_X', 'i', 'r', 'c', 'uu_X', 'ui_X', 'rui_X', 'iu_X', 'ii_X', 'comb', 'ri_X', 'ci_X']

np.random.seed([T, j, styles.index(style), int(mc), int(100*epsilon), \
    num_samples, trials])

null = True # dummy setting of null; doesn't matter

lengths = []
coverage = []
# iterate through trials
for job_ind in range(trials):
    counts = np.zeros(11)
    if job_ind % 10 == 0:
        print(job_ind)

    if type == 'conformal':
        test_stat = test_statistics.bandit_non_stationarity 
        if algo_name == 'epsilon_greedy':
            algorithm = bandit_non_stationarity.epsilon_greedy.EpsilonGreedy(T, epsilon, null, True if style=='c' else False)
        if algo_name == 'ucb':
            algorithm = bandit_non_stationarity.ucb.UCB(T, null, True if style=='c' else False)
    if type == 'confidence':
        test_stat = test_statistics.factored_bandit_distributional
        if algo_name == 'epsilon_greedy':
            algorithm = factored_bandit_distributional.epsilon_greedy.EpsilonGreedy(T, epsilon, null, True if style=='ci_X' else False, 4.) # b_0 = 4
        if algo_name == 'ucb':
            algorithm = factored_bandit_distributional.ucb.UCB(T, null, 4.) # b_0 = 4
    
    # get the true dataset
    true_data = algorithm.get_dataset()
    
    # iterate over b's:
    indexer = 0

    # depending on if conformal or confidence, b value range is different
    if type == 'conformal':
        last_action = true_data[-1][0]
        b_vals = np.linspace(-5, 5,11)
    if type == 'confidence':
        b_vals = np.linspace(-1,9,11)
    for b in b_vals:
        b_data = copy.deepcopy(true_data)
        
        if type == 'conformal':
            b_data[-1] = (b_data[-1][0],b) # transform just the last reward
            if mc:
                p_plus, p_minus, _ = mc_construct_rand_p_value(algorithm, b_data, test_stat, style, num_samples)
            else:
                p_plus, p_minus = mcmc_construct_rand_p_value(algorithm, b_data, test_stat, style, num_samples)

        if type == 'confidence':
            # transform data
            for index in range(len(true_data)):
                b_data[index] = (b_data[index][0],true_data[index][1] - b*int(true_data[index][0][0]==1 and true_data[index][0][1]==1))
            # obtain the p-values
            if mc:
                p_plus, p_minus = conf_interval_mc_construct_rand_p_value(algorithm, b_data, test_stat, style, b, num_samples)
            else:
                p_plus, p_minus = conf_interval_mcmc_construct_rand_p_value(algorithm, b_data, test_stat, style, b, num_samples)


        # calculate acceptance or not
        if p_plus > alpha:
            counts[indexer] = 1
        else:
            counts[indexer] = 0

        # calculate coverage for confidence
        if type == 'confidence':
            if b == 4:
                if p_plus > alpha:
                    coverage.append(1)
                else:
                    coverage.append(0)
        
        indexer += 1
    
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

    if type == 'conformal':
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
length_file_name = f'{type}_length/{algo_name}_{T}_{num_samples}_{j}_{mc}_{style}_{epsilon}'
coverage_file_name = f'{type}_coverage/{algo_name}_{T}_{num_samples}_{j}_{mc}_{style}_{epsilon}'

end = time.time()

elapsed_time = end-start


with open(length_file_name, 'w+') as f:
        json.dump(lengths, f)

# save time in the length file
with open(length_file_name + '_time', 'w+') as f:
    json.dump(elapsed_time, f)

with open(coverage_file_name, 'w+') as f:
        json.dump(coverage, f)

