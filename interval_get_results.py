import json
import sys
import numpy as np
import os


coverage_coverage_or_length = str(sys.argv[7])
type = str(sys.argv[6])
num_samples = int(sys.argv[5])
epsilon = float(sys.argv[4])
mc = eval(str(sys.argv[3]))
style = str(sys.argv[2])
algo_name = str(sys.argv[1])

alpha = 0.05
average = 0

prefix = type + '_' + coverage_coverage_or_length + '/' + algo_name + '_'

results = []
for T in [10,20,30,40,50,60,70,80,90,100]:
    total_counts = []
    for j in range(1,41):
        f =  prefix+str(T)+'_'+str(num_samples)+'_'+str(j)+'_'+str(mc)+'_'+str(style)+'_'+str(epsilon)
        if os.path.isfile(f):
            counts= json.load(open(f, 'r'))
            total_counts = total_counts + counts
        else:
            print(f)

    average = np.mean(total_counts)
    

    stderr = np.std(total_counts)/np.sqrt(len(total_counts))
    results.append((average,stderr))

print(results)