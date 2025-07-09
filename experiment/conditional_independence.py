import numpy as np
import pandas as pd
from collections import deque
import time
import sys
import copy
from sklearn import linear_model
pd.set_option('future.no_silent_downcasting', True)


from helpers import restricted_epsilon_greedy, mc_construct_rand_p_value,factored_bandit_distributional_score

job = int(sys.argv[1])
np.random.seed(job)

alpha=0.05

# Read in crop data
treatments = np.array(['T1', 'T2', 'T3', 'T4'])
treatment_mapping = {treatments[0]: 0, treatments[1]: 1,\
                      treatments[2]: 2, treatments[3]: 3}
crops = np.array(['Zea mays L.', 'Triticum aestivum L.', \
                  'Glycine max L.'])
crop_mapping = {crops[0]: 0, crops[1]: 1,\
                      crops[2]: 2,}

data = pd.read_csv('cleaned_agronomic_data.csv')[['year', 'treatment', 'crop', 'crop_only_yield_kg_ha']]

data['treatment'] = data['treatment'].replace(treatment_mapping)
data['crop'] = data['crop'].replace(crop_mapping)

data = data[data['treatment'].isin([0,1,2,3])]

years = data['year'].unique()

filtered_years = []
filtered_crops = []
filtered_actions = []
for year in years:
    for crop in (data[data['year'] == year])['crop'].unique():
        for action in range(4):
            if len(data.loc[(data['year'] == year) \
                            & (data['treatment'] == action) \
                            & (data['crop'] == crop), \
                            'crop_only_yield_kg_ha']) < 6:
                filtered_years.append(year)
                filtered_crops.append(crop)
                filtered_actions.append(action)

filtered = pd.DataFrame({
    "year": filtered_years,
    "crop": filtered_crops,
    'treatment': filtered_actions
})

filtered=filtered.sort_values(['year','crop'])
                


average_yields = (
    data.groupby(['year', 'treatment', 'crop'])['crop_only_yield_kg_ha']
    .mean()
    .reset_index()
)

average_yields=average_yields.sort_values(['year', 'crop'])


nulls = [
    [[0,1],[2],[3]],
    [[0,2],[1],[3]],
    [[0,3],[1],[2]],
    [[1,2],[0],[3]],
    [[1,3],[0],[2]],
    [[2,3],[0],[1]],
    [[0,1,2],[3]],
    [[3,1,2],[0]],
    [[0,3,2],[1]],
    [[0,1,3],[2]],
    [[0,1,2,3]]
]

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

epsilon=0.1

true_data =restricted_epsilon_greedy(average_yields, epsilon, conditional=False)
iid_data = restricted_epsilon_greedy(average_yields, 1., conditional=False)


print("----------------------------------")
print(f'Results for eps={epsilon}')

iid_results = [mc_construct_rand_p_value(iid_data, null_hypothesis_g, \
                                            factored_bandit_distributional_score, 'i_X', 1., \
                                            num_samples=1000) for null_hypothesis_g in nulls]

im_results = [mc_construct_rand_p_value(true_data, null_hypothesis_g, \
                                            factored_bandit_distributional_score, 'i_X', epsilon, \
                                                num_samples=1000) for null_hypothesis_g in nulls]



summary_df = pd.DataFrame({"iid p_plus": [result[0] for result in iid_results], \
                            "iid p_minus": [result[1] for result in iid_results], \
                        "im_X p_plus": [result[0] for result in im_results],\
                        "im_X p_minus": [result[1] for result in im_results],\
                            })

summary_df.index = null_names
print(summary_df)
summary_df.to_csv(f'im_X_experiment_results/eps_{int(100*epsilon)}_job{job}.csv', index=False)
print("----------------------------------")

imX_rejections = []
iid_rejections = []
for null_ind in range(11):
    im_result = im_results[null_ind]
    iid_result = iid_results[null_ind]

    imX_p_plus, imX_p_minus = im_result[:2]
    iid_p_plus, iid_p_minus = iid_result[:2]
    
    
    if imX_p_minus > alpha:
        imX_randomized_rejection = 0
    elif imX_p_plus > alpha:
        imX_randomized_rejection = 1-int(np.random.uniform() < (imX_p_plus - alpha)/(imX_p_plus - imX_p_minus))
    else:
        imX_randomized_rejection = 1


    if iid_p_minus > alpha:
        iid_randomized_rejection = 0
    elif iid_p_plus > alpha:
        iid_randomized_rejection = 1-int(np.random.uniform() < (iid_p_plus - alpha)/(iid_p_plus - iid_p_minus))
    else:
        iid_randomized_rejection = 1

    imX_rejections.append(imX_randomized_rejection)
    iid_rejections.append(iid_randomized_rejection)

imX_avg_power = np.mean(imX_rejections)
iid_avg_power = np.mean(iid_rejections)


power_results = [imX_avg_power, iid_avg_power]
with open(f"cond_indep_rejection_results/powers_j{job}.csv", "at") as file:
    file.write(",".join(map(str, power_results)) + "\n")
        