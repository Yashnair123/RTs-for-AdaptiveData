import json
import sys
import numpy as np
import os

output = []
algo_name = 'epsilon_greedy'
epsilon = 0.1
num_samples = 100
setting = 2
for style in ['us', 's']:
    for mc in [True]:
        for null in [False, True]:
            alpha = 0.05
            average = 0

            if setting == 0:
                prefix = 'bandit_non_stationarity/results/'
            if setting == 1:
                prefix = 'factored_bandit_distributional/results/'
            if setting == 2:
                prefix = 'contextual_bandit_distributional/results/'
            if setting == 3:
                prefix = 'mdp_non_stationarity/results/'
            if setting == 4:
                prefix = 'contextual_non_stationarity/results/'

            results = []
            for T in [10,20,30,40,50,60,70,80,90,100]:
                total_counts = []
                for j in range(1,41):
                    f =  prefix+algo_name+'_'+str(T)+'_'+str(num_samples)+'_'+str(j)+'_'+str(null)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+'_time'
                    if os.path.isfile(f):
                        elapsed_time = json.load(open(f, 'r'))
                        total_counts.append(elapsed_time)
                    else:
                        print(f)

                average = np.mean(total_counts)
                

                stderr = np.std(total_counts)/np.sqrt(len(total_counts))
                results.append((average/25.,stderr/5.))

            style_text = 'uniform_simulation' if style == 'us' else 'simulation'
            mc_or_mcmc = 'mc' if mc == True else 'mcmc'
            null_or_alternatives = 'nulls' if null == True else 'alternatives'
            o = mc_or_mcmc+'_'+algo_name+'_'+style_text+'_'+null_or_alternatives+'_times = ' + str(results)
            print(o)
            output.append(o)



algo_name = 'elinucb'
epsilon = 0.
num_samples = 100
setting = 2
for style in ['us']:
    for mc in [True]:
        for null in [False, True]:
            alpha = 0.05
            average = 0

            if setting == 0:
                prefix = 'bandit_non_stationarity/results/'
            if setting == 1:
                prefix = 'factored_bandit_distributional/results/'
            if setting == 2:
                prefix = 'contextual_bandit_distributional/results/'
            if setting == 3:
                prefix = 'mdp_non_stationarity/results/'
            if setting == 4:
                prefix = 'contextual_non_stationarity/results/'

            results = []
            for T in [10,20,30,40,50,60,70,80,90,100]:
                total_counts = []
                for j in range(1,41):
                    f =  prefix+algo_name+'_'+str(T)+'_'+str(num_samples)+'_'+str(j)+'_'+str(null)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+'_time'
                    if os.path.isfile(f):
                        elapsed_time = json.load(open(f, 'r'))
                        total_counts.append(elapsed_time)
                    else:
                        print(f)

                average = np.mean(total_counts)
                

                stderr = np.std(total_counts)/np.sqrt(len(total_counts))
                results.append((average/25.,stderr/5.))

            style_text = 'uniform_simulation' if style == 'us' else 'simulation'
            mc_or_mcmc = 'mc' if mc == True else 'mcmc'
            null_or_alternatives = 'nulls' if null == True else 'alternatives'
            o = mc_or_mcmc+'_linucb_'+style_text+'_'+null_or_alternatives+'_times = ' + str(results)
            print(o)
            output.append(o)
print()
print()
print()
for o in output:
    print(o)