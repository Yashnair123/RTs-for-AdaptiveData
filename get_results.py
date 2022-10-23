import json
import sys
import numpy as np
import os


setting = int(sys.argv[7])
num_samples = int(sys.argv[6])
epsilon = float(sys.argv[5])
mc = eval(str(sys.argv[4]))
style = str(sys.argv[3])
null = eval(str(sys.argv[2]))
algo_name = str(sys.argv[1])

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
    for j in range(1,201):
        f =  prefix+algo_name+'_'+str(T)+'_'+str(num_samples)+'_'+str(j)+'_'+str(null)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)
        if os.path.isfile(f):
            counts= json.load(open(f, 'r'))
            total_counts = total_counts + counts
        else:
            print(f)

    average = np.mean(total_counts)
    

    stderr = np.std(total_counts)/np.sqrt(len(total_counts))
    results.append((average,stderr))

print(results)