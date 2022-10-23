import numpy as np
import json
import sys
import copy

import contextual_bandit_distributional.elinucb

import test_statistics

from randomization_tests import conf_interval_mcmc_construct_rand_p_value, conf_interval_mc_construct_rand_p_value


num_samples = int(sys.argv[6])
epsilon = float(sys.argv[5])
mc = eval(str(sys.argv[4]))
style = str(sys.argv[3])
j = int(sys.argv[2])
T = int(sys.argv[1])

alpha = 0.05
styles = ['u', 's', 's1', 's2', 'uu', 'us', 'rus', 's1u', 's1s', 'c']
trials = 250

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

    test_stat = test_statistics.contextual_bandit_conditional_independence
    algorithm = contextual_bandit_distributional.elinucb.ELinUCB(T, epsilon, 2, null, 4.)
    
    # get the true dataset
    true_data = algorithm.get_dataset()

    # iterate over b's:
    indexer = 0
    for b in np.linspace(-1,9,11):
        b_data = copy.deepcopy(true_data)
        
        # transform data
        for index in range(len(true_data)):
            b_data[index][1] = true_data[index][1] - b*true_data[index][0][0]
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

        # calculate coverage
        if b == 4:
            if p_plus > alpha:
                coverage.append(1)
            else:
                coverage.append(0)
        
        indexer += 1
    
    # calculate length on the interval [-1,9] by using rounding of Chen, Chun, 
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
    

print(lengths)
print(np.mean(lengths), np.std(lengths)/np.sqrt(len(lengths)))
print(np.mean(coverage), np.std(coverage)/np.sqrt(len(coverage)))


# save file
length_file_name = f'length/{T}_{num_samples}_{j}_{mc}_{style}'
coverage_file_name = f'coverage/{T}_{num_samples}_{j}_{mc}_{style}'

with open(length_file_name, 'w+') as f:
        json.dump(lengths, f)

with open(coverage_file_name, 'w+') as f:
        json.dump(coverage, f)

