import json
import sys
import numpy as np
import os
import pickle


type = str(sys.argv[1])

if type == 'confidence':
    possibilities = [('epsilon_greedy', 1., 'uu_X'),
                     ('epsilon_greedy', 0.1, 'ui_X'),
                     ('epsilon_greedy', 0.1, 'rui_X'),
                     ('epsilon_greedy', 0.1, 'comb'),
                     ('ucb', 0.0, 'ui_X'),
                     ('ucb', 0.0, 'rui_X'),
                     ('ucb', 0.0, 'comb')]
if type == 'conformal':
    possibilities = [('epsilon_greedy', 1., 'u'),
                     ('epsilon_greedy', 0.1, 'u'),
                     ('epsilon_greedy', 0.1, 'i'),
                     ('epsilon_greedy', 0.1, 'r'),
                     ('epsilon_greedy', 0.1, 'c'),
                     ('ucb', 0.0, 'u'),
                     ('ucb', 0.0, 'i')]

num_samples = 100


alpha = 0.05
average = 0

for (algo_name, epsilon, style) in possibilities:
    for coverage_coverage_or_length in ['coverage', 'length']:
        for mc in [True, False]:
            prefix = type + '_' + coverage_coverage_or_length + '/' + algo_name + '_'

            results = []
            time_results = []
            for T in [10,20,30,40,50,60,70,80,90,100]:
                total_counts = []
                total_times = []
                for j in range(1,51):
                    f =  prefix+str(T)+'_'+str(num_samples)+'_'+str(j)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)
                    if os.path.isfile(f):
                        counts= json.load(open(f, 'r'))
                        total_counts = total_counts + counts
                    else:
                        print(f)

                    if coverage_coverage_or_length == 'length':
                        f =  prefix+str(T)+'_'+str(num_samples)+'_'+str(j)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+'_time'
                        if os.path.isfile(f):
                            time= json.load(open(f, 'r'))
                            total_times.append(time)
                        else:
                            print(f)

                if coverage_coverage_or_length == 'length':
                    average_time = np.mean(total_times)

                    stderr_time = np.std(total_times)/np.sqrt(len(total_times))
                    time_results.append((average_time,stderr_time))


                average = np.mean(total_counts)
                
                stderr = np.std(total_counts)/np.sqrt(len(total_counts))
                results.append((average,stderr))

            file_name = 'collated_interval_results/' + '_'+type+'_'+coverage_coverage_or_length+'_'+algo_name+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+'.pkl'
            with open(file_name, 'wb') as f:
                pickle.dump(results, f)

            if coverage_coverage_or_length == 'length':
                file_name = 'collated_interval_results/time' + '_'+type+'_'+coverage_coverage_or_length+'_'+algo_name+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)+'.pkl'
                with open(file_name, 'wb') as f:
                    pickle.dump(time_results, f)



if type == 'conformal':
    for (algo_name, epsilon, style) in possibilities:
        for coverage_coverage_or_length in ['coverage', 'length']:
            prefix = coverage_coverage_or_length +'_shared' + '/' + algo_name + '_'

            for num_samples in [10,100]:
                results = []
                time_results = []
                for T in [10,20,30,40,50,60,70,80,90,100]:
                    total_counts = []
                    total_times = []
                    for j in range(1,51):
                        f =  prefix+str(T)+'_'+str(num_samples)+'_'+str(j)+'_'+str(style)+'_'+str(epsilon)
                        if os.path.isfile(f):
                            counts= json.load(open(f, 'r'))
                            total_counts = total_counts + counts
                        else:
                            print(f)

                        if coverage_coverage_or_length == 'length':
                            f =  prefix+str(T)+'_'+str(num_samples)+'_'+str(j)+'_'+str(style)+'_'+str(epsilon)+'_time'
                            if os.path.isfile(f):
                                time = json.load(open(f, 'r'))
                                total_times = total_times + time
                            else:
                                print(f)

                    if coverage_coverage_or_length == 'length':
                        average_time = np.mean(total_times)

                        stderr_time = np.std(total_times)/np.sqrt(len(total_times))
                        time_results.append((average_time,stderr_time))


                    average = np.mean(total_counts)
                    
                    stderr = np.std(total_counts)/np.sqrt(len(total_counts))
                    results.append((average,stderr))

                    


                file_name = 'collated_interval_results/_share' + '_'+type+'_'+coverage_coverage_or_length+'_'+str(num_samples)+'_'+algo_name+'_'+str(style)+'_'+str(epsilon)+'.pkl'
                with open(file_name, 'wb') as f:
                    pickle.dump(results, f)

                if coverage_coverage_or_length == 'length':
                    file_name = 'collated_interval_results/time_share' + '_'+type+'_'+coverage_coverage_or_length+'_'+str(num_samples)+'_'+algo_name+'_'+str(style)+'_'+str(epsilon)+'.pkl'
                    with open(file_name, 'wb') as f:
                        pickle.dump(time_results, f)