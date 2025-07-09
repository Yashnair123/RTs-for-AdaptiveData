import pandas as pd
import numpy as np
from tqdm import tqdm


alpha = 0.05
epsilon=0.1
num_trials = 500

method_p_value_names = [
                        "uniform iid",
                        r"$\text{imitation}_X$"]
null_names = [
"T1=T2",
"T1=T3",
"T1=T4",
"T2=T3",
"T2=T4",
"T3=T4",
"T1=T2=T3",
"T2=T3=T4",
"T1=T3=T4",
"T1=T2=T4",
'T1=T2=T3=T4'
]

method_to_column_indexer_p_plus = [0,2]
method_to_column_indexer_p_minus = [1,3]


method_arrays = []
null_arrays = []

reject_arrays = dict()
randomized_reject_arrays = dict()
ess_arrays = dict()

for method_indexer in range(2):
    for index in range(11):
        reject_arrays[(method_indexer, index)] = []
        randomized_reject_arrays[(method_indexer, index)] = []


for job in range(500):
    df = pd.read_csv(f'im_X_experiment_results/eps_{int(100*epsilon)}_job{job}.csv')
    for method_indexer in range(2):    
        for index, row in df.iterrows():
            p_plus = row.iloc[method_to_column_indexer_p_plus[method_indexer]]
            p_minus = row.iloc[method_to_column_indexer_p_minus[method_indexer]]

            method_arrays.append(method_p_value_names[method_indexer])
            null_arrays.append(null_names[index])
            
            reject_arrays[(method_indexer, index)].append(int(p_plus <= 0.05))
            if p_minus > alpha:
                randomized_rejection = 0
            elif p_plus > alpha:
                randomized_rejection = 1-int(np.random.uniform() < (p_plus - alpha)/(p_plus - p_minus))
            else:
                randomized_rejection = 1
            randomized_reject_arrays[(method_indexer, index)].append(randomized_rejection)

# Create an empty DataFrame with string values
table_data = pd.DataFrame(index=null_names, columns=method_p_value_names)

# Fill the DataFrame with "$mean \pm margin$" formatted entries
for method_indexer, method_name in enumerate(method_p_value_names):
    for null_index, null_name in enumerate(null_names):
        values = np.array(randomized_reject_arrays[(method_indexer, null_index)])
        mean_val = np.mean(values)
        std_err = np.std(values, ddof=1) / np.sqrt(len(values))
        margin = 1.96 * std_err
        table_data.loc[null_name, method_name] = f"${mean_val:.3f} \\pm {margin:.3f}$"


# Save to LaTeX
latex_str = table_data.to_latex(index=True, column_format='lcc', escape=False)
with open("latex_tables/randomized_rejections_table.tex", "w") as f:
    f.write(latex_str)


# Create a one-row DataFrame for averages
avg_table_data = pd.DataFrame(index=["Average"], columns=method_p_value_names)

# Format average row with LaTeX math mode and \pm
for method_indexer, method_name in enumerate(method_p_value_names):
    all_vals = []
    for null_index in range(len(null_names)):
        all_vals.extend(randomized_reject_arrays[(method_indexer, null_index)])
    
    all_vals = np.array(all_vals)
    mean_val = np.mean(all_vals)
    std_err = np.std(all_vals, ddof=1) / np.sqrt(len(all_vals))
    margin = 1.96 * std_err
    avg_table_data.loc["Average", method_name] = f"${mean_val:.3f} \\pm {margin:.3f}$"

# Save to LaTeX
latex_str_avg = avg_table_data.to_latex(index=True, column_format='lcc', escape=False)
with open("latex_tables/randomized_rejections_table_avg.tex", "w") as f:
    f.write(latex_str_avg)