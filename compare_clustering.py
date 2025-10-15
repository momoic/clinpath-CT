#sys.path.append(os.path.dirname(os.path.abspath(__name__)))

import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import stepmix
import lifelines
from lifelines.statistics import logrank_test
from cluster_functions import calc_sig
from cluster_functions import calc_sig_MSS
from cluster_functions import risk_class_MSS
import seaborn as sns
import pickle 


data = pd.read_csv("C:\\PhD work\\clinpath_CT\\SEER_final.csv")

data = data.drop(columns=["Unnamed: 0", "Stage"])

#data = data.drop(columns=["Node_Status", "MSS", "MSS_Time"])
data_gmm = data.copy()
scaler = sk.preprocessing.StandardScaler()
data_ml = pd.DataFrame(scaler.fit_transform(data_gmm), columns = data_gmm.columns)

"""
index is 11 for SEER_final data
bic_scores = []

for i in range(1, 50):
    gmm = sk.mixture.GaussianMixture(n_components=i, covariance_type='full', random_state = 42, max_iter = 100)
    gmm.fit(data_ml)
    bic_scores.append(gmm.bic(data_ml))

#plt.plot(range(1, 50), bic_scores, marker='o')
#plt.show()

diff_list = []
for i in range(1, len(bic_scores)-3):
    diff_list.append((bic_scores[i+3] - bic_scores[i])/bic_scores[i])

index = [i for i, d in enumerate(diff_list) if d<0.10][1]
"""
index = 11
gmm = sk.mixture.GaussianMixture(n_components=index, covariance_type='full', random_state = 42, max_iter = 100)
gmm.fit(data_ml)

data_gmm["Group"] = gmm.predict(data_ml)
data_ml["Group"] = data_gmm["Group"]

data2 = calc_sig_MSS(data_gmm, index)
risk_data = risk_class_MSS(data_gmm, data2)

print(sum(risk_data["MSS"]))
print(sum(risk_data.query("Risk == 0")["MSS"]))
print(sum(risk_data.query("Risk == 2")["MSS"]))

data_sm = pd.DataFrame(data.copy())

factor_cols = [col for col in data.columns if data[col].nunique() == 2]
continuous_cols = [col for col in data.columns if col not in factor_cols]

mixed_data, mixed_descriptor = stepmix.utils.get_mixed_descriptor(
    dataframe  = data_sm,
    continuous = continuous_cols,
    binary = factor_cols
)
index = 4
model = stepmix.StepMix(n_components= index, measurement = mixed_descriptor, random_state=42)
model.fit(mixed_data)
data_sm["Group"] = model.predict(mixed_data)
pd.crosstab(data_sm["Group"], data_gmm["Group"])
sk.metrics.adjusted_rand_score(data_sm["Group"], data_gmm["Group"])
data2_sm = calc_sig_MSS(data_sm, index)
risk_data_sm = risk_class_MSS(data_sm, data2_sm)
print(data2_sm)
print(risk_data_sm.groupby("Group").size())
print(risk_data_sm.groupby("Risk").size())
pd.crosstab(risk_data_sm["Risk"], risk_data["Risk"])
sk.metrics.adjusted_rand_score(risk_data_sm["Risk"], risk_data["Risk"])

print(sum(risk_data_sm["MSS"]))
print(sum(risk_data_sm.query("Risk == 0")["MSS"]))
print(sum(risk_data_sm.query("Risk == 2")["MSS"]))


differences = risk_data_sm.compare(risk_data, keep_equal=True, keep_shape=True, align_axis=1)

differences = differences.iloc[:, (differences.columns.get_level_values(1) == 'self') | (differences.columns.get_level_values(0) == 'Risk')]
differences[('Risk', 'self')]
differences.loc[differences[('Risk', 'self')] != differences[('Risk', 'other')], :]
#p_values = stepmix.bootstrap.blrt_sweep(model, mixed_data, low = 1, high = 5, n_repetitions = 1, random_state = 42)

"""grid = {
    'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'n_steps' : [1, 2, 3]
}
gs = sk.model_selection.GridSearchCV(estimator=model, cv=3, param_grid=grid)
gs.fit(mixed_data)
results = pd.DataFrame(gs.cv_results_)
print(results[['param_n_components', 'param_n_steps', 'mean_test_score']])
results["Val. Log Likelihood"] = results['mean_test_score']
sns.set_style("darkgrid")
sns.lineplot(data=results, x='param_n_components', y='Val. Log Likelihood',
             hue='param_n_steps', palette='Dark2')
plt.show()"""

