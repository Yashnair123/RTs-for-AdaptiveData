import json
import sys
import numpy as np
import os

setting = 4
num_samples = 100


algo_name = 'epsilon_greedy'
epsilon = 0.1
for style in ['u', 's1', 's2', 's3']:
    for null in [True, False]:
        for mc in [True]:
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
                    f =  prefix+algo_name+'_'+str(T)+'_'+str(num_samples)+'_'+str(j)+'_'+str(null)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+'_eff_samples'
                    if os.path.isfile(f):
                        counts= json.load(open(f, 'r'))
                        total_counts = total_counts + counts
                    else:
                        print(f)

                average = np.mean(total_counts)
                

                stderr = np.std(total_counts)/np.sqrt(len(total_counts))
                results.append((average,stderr))

            if style == 'u':
                style_text = 'uniform'
            elif style == 's1':
                style_text = 'simulation1'
            elif style == 's2':
                style_text = 'simulation2'
            else:
                style_text = 'simulation3'


            mc_or_mcmc = 'mc' if mc == True else 'mcmc'

            suffix = 'nulls' if null == True else 'alternatives'

            o = 'eff_'+mc_or_mcmc+'_'+algo_name+'_'+style_text+'_'+suffix + ' = ' + str(results)
            print(o)



algo_name = 'elinucb'
epsilon = 0.
for style in ['u', 's1']:
    for null in [True, False]:
        for mc in [True]:
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
                    f =  prefix+algo_name+'_'+str(T)+'_'+str(num_samples)+'_'+str(j)+'_'+str(null)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+'_eff_samples'
                    if os.path.isfile(f):
                        counts= json.load(open(f, 'r'))
                        total_counts = total_counts + counts
                    else:
                        print(f)

                average = np.mean(total_counts)
                

                stderr = np.std(total_counts)/np.sqrt(len(total_counts))
                results.append((average,stderr))

            if style == 'u':
                style_text = 'uniform'
            elif style == 's1':
                style_text = 'simulation1'
            elif style == 's2':
                style_text = 'simulation2'
            else:
                style_text = 'simulation3'


            mc_or_mcmc = 'mc' if mc == True else 'mcmc'

            suffix = 'nulls' if null == True else 'alternatives'

            o = 'eff_'+mc_or_mcmc+'_linucb_'+style_text+'_'+suffix + ' = ' + str(results)
            print(o)