import numpy as np
import json

prefix = './robustness_results/'

epsilons = np.linspace(0.1,0.9,9)
alpha=0.05

type_i_errs = []
type_i_err_sds = []

ubs = []
ub_sds = []
for analysis_eps in epsilons:
        for style in ['u', 'i', 'r']:
            for true_eps in epsilons:
                counts = []
                inflations = []
                for block in range(4):
                    block_counts = json.load(open(prefix+f'counts_egreedy_{style}_{analysis_eps}_{true_eps}_{block}', 'r'))
                    block_inflations = json.load(open(prefix+f'inflations_egreedy_{style}_{analysis_eps}_{true_eps}_{block}', 'r'))

                    counts += block_counts
                    inflations += block_inflations

                type_i = (np.mean(counts))
                type_i_se = (np.std(counts)/np.sqrt(len(counts)))

                ub = (alpha+np.mean(inflations))
                ub_se = (np.std(inflations)/np.sqrt(len(inflations)))


                file_name=f'collated_robustness_results/egreedy_{style}_{analysis_eps}_{true_eps}'
                with open(file_name, 'w+') as f:
                    json.dump([type_i, type_i_se, ub, ub_se], f)