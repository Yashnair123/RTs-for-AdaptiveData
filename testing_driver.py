import numpy as np
import json
import sys
import matplotlib.pyplot as plt

import bandit_non_stationarity.epsilon_greedy, bandit_non_stationarity.ucb, \
    factored_bandit_distributional.epsilon_greedy, factored_bandit_distributional.ucb, \
        contextual_bandit_distributional.elinucb, contextual_bandit_distributional.epsilon_greedy,\
            mdp_non_stationarity.epsilon_greedy, contextual_non_stationarity.elinucb, \
                contextual_non_stationarity.epsilon_greedy

import test_statistics

import time

from randomization_tests import mcmc_construct_rand_p_value, mc_construct_rand_p_value

start = time.time()

setting = int(sys.argv[10])
num_samples = int(sys.argv[9])
epsilon = float(sys.argv[8])
mc = eval(str(sys.argv[7]))
style = str(sys.argv[6])
null = eval(str(sys.argv[5]))
algo_name = str(sys.argv[4])
j = int(sys.argv[3])
T = int(sys.argv[2])
trials = int(sys.argv[1])

alpha = 0.05
styles = ['u', 'i_X', 'i', 'r', 'c', 'uu_X', 'ui_X', 'rui_X', 'iu_X', 'ii_X', 'comb', 'ri_X', 'ci_X']

np.random.seed([setting, T, j, int(algo_name[0] == 'e'), int(null), styles.index(style), int(mc), int(100*epsilon), \
    num_samples, trials])

resamples = []

eff_sample_ratios = []

counts = []
# iterate through trials
for job_ind in range(trials):
    if job_ind % 10 == 0:
        print(job_ind)
    # obtain the correct algorithm
    if setting == 0:
        test_stat = test_statistics.bandit_non_stationarity
        file_name = f'bandit_non_stationarity/results/{algo_name}_{T}_{num_samples}_{j}_{null}_{mc}_{style}_{epsilon}'
        if algo_name == 'epsilon_greedy':
            algorithm = bandit_non_stationarity.epsilon_greedy.EpsilonGreedy(T, epsilon, null, True if style=='c' else False)
        if algo_name == 'ucb':
            algorithm = bandit_non_stationarity.ucb.UCB(T, null, True if style=='c' else False)
    if setting == 1:
        test_stat = test_statistics.factored_bandit_distributional
        file_name = f'factored_bandit_distributional/results/{algo_name}_{T}_{num_samples}_{j}_{null}_{mc}_{style}_{epsilon}'
        if algo_name == 'epsilon_greedy':
            algorithm = factored_bandit_distributional.epsilon_greedy.EpsilonGreedy(T, epsilon, null, True if style=='ci_X' else False)
        if algo_name == 'ucb':
            algorithm = factored_bandit_distributional.ucb.UCB(T, null)
    if setting == 2:
        test_stat = test_statistics.contextual_bandit_conditional_independence
        file_name = f'contextual_bandit_distributional/results/{algo_name}_{T}_{num_samples}_{j}_{null}_{mc}_{style}_{epsilon}'
        if algo_name == 'elinucb':
            algorithm = contextual_bandit_distributional.elinucb.ELinUCB(T, epsilon, 2, null)
        if algo_name == 'epsilon_greedy':
            algorithm = contextual_bandit_distributional.epsilon_greedy.EpsilonGreedy(T, epsilon, 2, null)
    if setting == 3:
        test_stat = test_statistics.mdp_non_stationarity
        file_name = f'mdp_non_stationarity/results/{algo_name}_{T}_{num_samples}_{j}_{null}_{mc}_{style}_{epsilon}'
        if algo_name == 'epsilon_greedy':
            algorithm = mdp_non_stationarity.epsilon_greedy.EpsilonGreedy(T, epsilon, null, True if style=='c' else False)
    if setting == 4:
        test_stat = test_statistics.contextual_bandit_non_stationarity
        file_name = f'contextual_non_stationarity/results/{algo_name}_{T}_{num_samples}_{j}_{null}_{mc}_{style}_{epsilon}'
        if algo_name == 'elinucb':
            algorithm = contextual_non_stationarity.elinucb.ELinUCB(T, epsilon, 100, null, True if style=='c' else False)
        if algo_name == 'epsilon_greedy':
            algorithm = contextual_non_stationarity.epsilon_greedy.EpsilonGreedy(T, epsilon, 100, null, True if style=='c' else False)
    
    # get the true dataset
    true_data = algorithm.get_dataset()

    # obtain the p-values
    if mc:
        p_plus, p_minus, eff_sample_ratio = mc_construct_rand_p_value(algorithm, true_data, test_stat, style, num_samples)
        eff_sample_ratios.append(eff_sample_ratio)
    else:
        p_plus, p_minus = mcmc_construct_rand_p_value(algorithm, true_data, test_stat, style, num_samples)


    # calculate the randomized p-value
    if p_minus > alpha:
        counts.append(0)
    elif p_plus > alpha:
        counts.append(1-int(np.random.uniform() < (p_plus - alpha)/(p_plus - p_minus)))
    else:
        counts.append(1)

end = time.time()

# print the counts and mean and sd
print(counts)
print(np.mean(counts), np.std(counts)/np.sqrt(len(counts)))


# save file
with open(file_name, 'w+') as f:
    json.dump(counts, f)


elapsed_time = end-start
# save time elapsed
with open(file_name + '_time', 'w+') as f:
    json.dump(elapsed_time, f)

if mc:
    file_name = file_name + '_eff_samples'
    # save file
    with open(file_name, 'w+') as f:
        json.dump(eff_sample_ratios, f)


