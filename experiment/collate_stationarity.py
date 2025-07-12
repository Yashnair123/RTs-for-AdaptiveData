import pandas as pd
import numpy as np
from tqdm import tqdm
import sys



alpha = 0.05
epsilon=0.1
num_trials = 500

method_p_value_names = ['uniform iid',
                        r"$\text{imitation}_{\pi}$",
                        r"$\text{re-imitation}_{\pi}$",
                        r"$\text{cond-imitation}_{\pi}$"]

method_to_column_indexer_p_plus = [0, 2, 4, 6]
method_to_column_indexer_p_minus = [1,3,5,7]


method_arrays = []
null_arrays = []

reject_arrays = dict()
randomized_reject_arrays = dict()


for method_indexer in range(6):
    randomized_reject_arrays[(method_indexer)] = []

for job in range(500):
    df = pd.read_csv(f'stationarity_results/eps_{int(100*epsilon)}_job{job}.csv')
    for method_indexer in range(4):    
        for index, row in df.iterrows():
            p_plus = row.iloc[method_to_column_indexer_p_plus[method_indexer]]
            p_minus = row.iloc[method_to_column_indexer_p_minus[method_indexer]]

            method_arrays.append(method_p_value_names[method_indexer])
            
            if p_minus > alpha:
                randomized_rejection = 0
            elif p_plus > alpha:
                randomized_rejection = 1-int(np.random.uniform() < (p_plus - alpha)/(p_plus - p_minus))
            else:
                randomized_rejection = 1
            randomized_reject_arrays[(method_indexer)].append(randomized_rejection)



# Construct one-row summary table
avg_table_data = pd.DataFrame(index=["Average"], columns=method_p_value_names)

for method_indexer, method_name in enumerate(method_p_value_names):
    values = np.array(randomized_reject_arrays[method_indexer])
    mean_val = np.mean(values)
    std_err = np.std(values, ddof=1) / np.sqrt(len(values))
    margin = 1.96 * std_err
    avg_table_data.loc["Average", method_name] = f"${mean_val:.3f}$ $(\\pm {margin:.3f})$"

# Save to LaTeX
latex_str = avg_table_data.to_latex(index=True, column_format='lcccc', escape=False)
with open("latex_tables/stationarity_randomized_rejections_avg.tex", "w") as f:
    f.write(latex_str)