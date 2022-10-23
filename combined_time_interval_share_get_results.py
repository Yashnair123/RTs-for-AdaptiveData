import json
import sys
import numpy as np
import os

output = []


algo_name = 'epsilon_greedy'
epsilon = 1.

for num_samples in [10,100]:
    for style in ['u']:
        alpha = 0.05
        average = 0

        prefix = 'length_shared' + '/' + algo_name + '_'

        results = []
        for T in [10,20,30,40,50,60,70,80,90,100]:
            total_counts = []
            for j in range(1,41):
                f =  prefix+str(T)+'_'+str(num_samples)+'_'+str(j)+'_'+str(style)+'_'+str(epsilon)+'_time'
                if os.path.isfile(f):
                    time = json.load(open(f, 'r'))
                    total_counts.append(time)
                else:
                    print(f)

            average = np.mean(total_counts)


            stderr = np.std(total_counts)/np.sqrt(len(total_counts))
            results.append((average/25.,stderr/5.))

        mc_or_mcmc = 'mc'
        o = 'mc_iid_times_'+str(num_samples)+' = ' + str(results)
        print(o)
        output.append(o)



algo_name = 'epsilon_greedy'
epsilon = 0.1
for num_samples in [10,100]:
    for style in ['u', 's1', 's2', 's3']:
        alpha = 0.05
        average = 0

        prefix = 'length_shared' + '/' + algo_name + '_'

        results = []
        for T in [10,20,30,40,50,60,70,80,90,100]:
            total_counts = []
            for j in range(1,41):
                f =  prefix+str(T)+'_'+str(num_samples)+'_'+str(j)+'_'+str(style)+'_'+str(epsilon)+'_time'
                if os.path.isfile(f):
                    time = json.load(open(f, 'r'))
                    total_counts.append(time)
                else:
                    print(f)

            average = np.mean(total_counts)


            stderr = np.std(total_counts)/np.sqrt(len(total_counts))
            results.append((average/25.,stderr/5.))

            
        if style == 'u':
            style_text = 'uniform'
        elif style == 's1':
            style_text = 'simulation1'
        elif style == 's2':
            style_text = 'simulation2'
        else:
            style_text = 'simulation3'

        mc_or_mcmc = 'mc'

        o = mc_or_mcmc+'_'+algo_name+'_'+style_text+'_times_'+str(num_samples)+' = ' + str(results)
        print(o)
        output.append(o)


algo_name = 'ucb'
epsilon = 0.
for num_samples in [10,100]:
    for style in ['u', 's1']:
        alpha = 0.05
        average = 0

        prefix = 'length_shared' + '/' + algo_name + '_'

        results = []
        for T in [10,20,30,40,50,60,70,80,90,100]:
            total_counts = []
            for j in range(1,41):
                f =  prefix+str(T)+'_'+str(num_samples)+'_'+str(j)+'_'+str(style)+'_'+str(epsilon)+'_time'
                if os.path.isfile(f):
                    time = json.load(open(f, 'r'))
                    total_counts.append(time)
                else:
                    print(f)

            average = np.mean(total_counts)


            stderr = np.std(total_counts)/np.sqrt(len(total_counts))
            results.append((average/25.,stderr/5.))

            
        if style == 'u':
            style_text = 'uniform'
        elif style == 's1':
            style_text = 'simulation1'
        elif style == 's2':
            style_text = 'simulation2'
        else:
            style_text = 'simulation3'

        mc_or_mcmc = 'mc'

        o = mc_or_mcmc+'_'+algo_name+'_'+style_text+'_times_'+str(num_samples)+' = ' + str(results)
        print(o)
        output.append(o)