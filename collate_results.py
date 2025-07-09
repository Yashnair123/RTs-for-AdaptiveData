import json
import sys
import numpy as np
import os
import pickle
from tqdm import tqdm


setting = int(sys.argv[1])

alpha = 0.05

suffix=''
if setting == 0:
    prefix = 'bandit_non_stationarity/'
    possibilities = [('epsilon_greedy', 'u', 1.),
                     ('epsilon_greedy', 'u', 0.1),
                     ('epsilon_greedy', 'i', 0.1),
                     ('epsilon_greedy', 'r', 0.1),
                     ('epsilon_greedy', 'c', 0.1),
                     ('ucb', 'u', 0.),
                     ('ucb', 'i', 0.)]
if setting == 1:
    prefix = 'factored_bandit_distributional/'
    possibilities = [('epsilon_greedy', 'uu_X', 1.),
                     ('epsilon_greedy', 'ui_X', 0.1),
                     ('epsilon_greedy', 'rui_X', 0.1),
                     ('epsilon_greedy', 'comb', 0.1),
                     ('ucb', 'ui_X', 0.),
                     ('ucb', 'rui_X', 0.),
                     ('ucb', 'comb', 0.)]
if setting == 2:
    prefix = 'contextual_bandit_distributional/'
    possibilities = [('epsilon_greedy', 'uu_X', 1.),
                     ('epsilon_greedy', 'i_X', 0.1),
                     ('epsilon_greedy', 'ui_X', 0.1),
                     ('elinucb', 'ui_X', 0.),
                     ('elinucb', 'i_X', 0.)]
if setting == 3:
    prefix = 'mdp_non_stationarity/'
    possibilities = [('epsilon_greedy', 'u', 1.),
                     ('epsilon_greedy', 'u', 0.1),
                     ('epsilon_greedy', 'i', 0.1),
                     ('epsilon_greedy', 'r', 0.1),
                     ('epsilon_greedy', 'c', 0.1),
                     ('epsilon_greedy', 'u', 0.),
                     ('epsilon_greedy', 'i', 0.)]
if setting == 4:
    prefix = 'contextual_non_stationarity/'
    possibilities = [('epsilon_greedy', 'u', 1.),
                     ('epsilon_greedy', 'u', 0.1),
                     ('epsilon_greedy', 'i', 0.1),
                     ('epsilon_greedy', 'r', 0.1),
                     ('epsilon_greedy', 'c', 0.1),
                     ('elinucb', 'u', 0.),
                     ('elinucb', 'i', 0.),
                     ('biased_iid', 'u', 0.1)]
if setting == 5:
    prefix='factored_bandit_more_arms/'
    possibilities = [('epsilon_greedy', 'uu_X', 1.),
                     ('epsilon_greedy', 'ui_X', 0.1),
                     ('epsilon_greedy', 'rui_X', 0.1),
                     ('epsilon_greedy', 'comb', 0.1),
                     ('ucb', 'ui_X', 0.),
                     ('ucb', 'rui_X', 0.),
                     ('ucb', 'comb', 0.)]
if setting == 6:
    prefix = 'contextual_bandit_distributional_g/'
    suffix='_2'
    possibilities = [('epsilon_greedy', 'uu_X', 1.),
                     ('ec', 'ui_X', 0.),
                     ('ec', 'rui_X', 0.),
                     ('ec', 'comb', 0.)]


for (algo_name, style, epsilon) in tqdm(possibilities):
    for null in [True, False]:
        for mc in [True, False]:
            returns = []
            if not null:
                if ((setting == 2 and style != 'uu_X') or setting != 2) and algo_name != 'biased_iid':
                    for num_samples in [100, 1000, 10000]:
                        total_counts = []
                        if setting not in [2,4,6]:
                            j_range = range(1,21) if num_samples==100 else range(1, 101)
                        else:
                            j_range = range(1,21) if num_samples==100 else range(1, 1001)

                        for j in j_range:
                            f =  prefix+'results/'+algo_name+'_'+str(100)+'_'+str(num_samples)+'_'+str(j)+'_'+str(null)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+suffix
                            if os.path.isfile(f):
                                counts= json.load(open(f, 'r'))
                                total_counts = total_counts + counts
                            else:
                                print(f)

                        average = np.mean(total_counts)
                        stderr = np.std(total_counts)/np.sqrt(len(total_counts))

                        returns.append((average, stderr))

                    # Write to file
                    file_name = prefix+'collated_results/'+algo_name+'_'+str(null)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+'varying_m' + '.pkl'
                    with open(file_name, 'wb') as f:
                        pickle.dump(returns, f)

            num_samples = 100
            average = 0
            results = []
            results_eff = []
            results_time = []
            for T in [10,20,30,40,50,60,70,80,90,100]:
                total_counts = []
                total_effs = []
                total_times = []
                for j in range(1,21):
                    f =  prefix+'results/'+algo_name+'_'+str(T)+'_'+str(num_samples)+'_'+str(j)+'_'+str(null)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+suffix
                    if os.path.isfile(f):
                        counts= json.load(open(f, 'r'))
                        total_counts = total_counts + counts
                    else:
                        print(f)
                    if mc:
                        eff_f =  prefix+'results/'+algo_name+'_'+str(T)+'_'+str(num_samples)+'_'+str(j)+'_'+str(null)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+suffix+'_eff_samples'
                        if os.path.isfile(eff_f):
                            effs = json.load(open(eff_f, 'r'))
                            total_effs = total_effs + effs
                        else:
                            print(eff_f)
                    time_f =  prefix+'results/'+algo_name+'_'+str(T)+'_'+str(num_samples)+'_'+str(j)+'_'+str(null)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+suffix+'_time'
                    if os.path.isfile(time_f):
                        times = json.load(open(time_f, 'r'))
                        total_times = total_times + times
                    else:
                        print(time_f)

                average = np.mean(total_counts)
                if mc:
                    average_eff = np.mean(total_effs)
                average_time = np.mean(total_times)

                stderr = np.std(total_counts)/np.sqrt(len(total_counts))
                if mc:
                    stderr_eff = np.std(total_effs)/np.sqrt(len(total_effs))
                stderr_time = np.std(total_times)/np.sqrt(len(total_times))

                results.append((average,stderr))
                if mc:
                    results_eff.append((average_eff,stderr_eff))
                results_time.append((average_time,stderr_time))

            # Write to file
            file_name = prefix+'collated_results/'+algo_name+'_'+str(num_samples)+'_'+str(null)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+'.pkl'
            with open(file_name, 'wb') as f:
                pickle.dump(results, f)

            if mc:
                # Write to file
                file_name = prefix+'collated_results/'+algo_name+'_'+str(num_samples)+'_'+str(null)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+'eff.pkl'
                with open(file_name, 'wb') as f:
                    pickle.dump(results_eff, f)

            # Write to file
            file_name = prefix+'collated_results/'+algo_name+'_'+str(num_samples)+'_'+str(null)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+'time.pkl'
            with open(file_name, 'wb') as f:
                pickle.dump(results_time, f)


if setting == 6:
    num_samples=100
    T=100
    null=False
    
    for (algo_name, style, epsilon) in tqdm(possibilities):
        for mc in [True,False]:
            results = []
            for dim in range(2,51):
                total_counts = []
                for j in range(1,21):
                    f =  prefix+'results/'+algo_name+'_'+str(100)+'_'+str(num_samples)+'_'+str(j)+'_'+str(null)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+'_'+str(dim)
                    if os.path.isfile(f):
                        counts= json.load(open(f, 'r'))
                        total_counts = total_counts + counts
                    else:
                        print(f)

                average = np.mean(total_counts)
                stderr = np.std(total_counts)/np.sqrt(len(total_counts))

                results.append((average,stderr))
            
            # Write to file
            file_name = prefix+'collated_results/varying_dim'+algo_name+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+'.pkl'
            with open(file_name, 'wb') as f:
                pickle.dump(results, f)