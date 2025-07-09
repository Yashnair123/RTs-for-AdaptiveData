import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys

pd.set_option('future.no_silent_downcasting', True)


from helpers import restricted_epsilon_greedy, mc_nonstationarity_construct_rand_p_value,\
    r2_stationarity_score

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


iid_rejections = []
imitation_rejections = []
reimitation_rejections = []
condimitation_rejections = []
epsilon=0.1

true_data, true_coinflips = restricted_epsilon_greedy(average_yields,epsilon,conditional=True)
iid_data, iid_coinflips = restricted_epsilon_greedy(average_yields,1.,conditional=True)

print("----------------------------------")
print(f'Results for eps={epsilon}')

iid_result = mc_nonstationarity_construct_rand_p_value(iid_data, \
                                            r2_stationarity_score, 'i', 1., \
                                                num_samples=1000)
imitation_result = mc_nonstationarity_construct_rand_p_value(true_data, \
                                            r2_stationarity_score, 'i', epsilon, \
                                                num_samples=1000)
reimitation_result = mc_nonstationarity_construct_rand_p_value(true_data, \
                                            r2_stationarity_score, 'r', epsilon, \
                                                num_samples=1000)
condimitation_result = mc_nonstationarity_construct_rand_p_value(true_data, \
                                            r2_stationarity_score, 'c', epsilon, \
                                                num_samples=1000, coin_flips=true_coinflips)



summary_df = pd.DataFrame({"iid p_plus": [iid_result[0]], \
                            "iid p_minus": [iid_result[1]], \
                        "imitation p_plus": [imitation_result[0]],\
                        "imitation p_minus": [imitation_result[1]],\
                        "reimitation p_plus": [reimitation_result[0]],\
                        "reimitation p_minus": [reimitation_result[1]],\
                        "condimitation p_plus": [condimitation_result[0]],\
                        "condimitation p_minus": [condimitation_result[1]]})

print(summary_df)
summary_df.to_csv(f'r2_stationarity_results/eps_{int(100*epsilon)}_job{job}.csv', index=False)
print("----------------------------------")