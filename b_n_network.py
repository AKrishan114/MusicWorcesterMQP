import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import math
from pomegranate import *
import pybbn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import itertools

new_d1 = pd.read_csv('211Clustering - LatestMetrics.csv')
new_d2 = pd.read_csv('anon_Patrons20250210 - anon_Patrons.csv.csv')
new_d1 = new_d1[new_d1.Segment != 'Best']
new_d1 = new_d1[new_d1.Segment != 'One&Done']
new_d1 = new_d1[new_d1.Segment != 'New']
new_d1 = new_d1[new_d1.Segment != 'High']
new_d1 = new_d1[new_d1.Subscriber != 'Current']

#print(new_d1[['Lifespan', 'Regularity', 'AYMGSForcast', 'PreferenceConfidence']].describe())


nd1 = new_d1.filter(items=['Lifespan','AYMGSForcast', 'Regularity', 'PreferenceConfidence', 'Cluster KMeans',])
nd2 = new_d2.filter(items=['EventClassPreferenceConfidence'])

new_data = nd1.merge(nd2, left_on='Cluster KMeans', right_on='EventClassPreferenceConfidence')

raw_d = pd.read_csv('/Users/natekaalman/Documents/Music Worcester Work/output1.csv')
raw_d2 = pd.read_csv('Copy of anon_PatronDetails - anon_PatronDetails.csv.csv')

d1 = raw_d.filter(items=['Lifespan','AYMGSForcast', 'EngagementConsistency', 'Entropy', 'Cluster KMeans',
                              'Cluster Agglo'], axis=1)

d2 = raw_d2.filter(items=['Regularity', 'PreferenceConfidence','Segment'], axis=1)

clean_d = d1.merge(d2, left_on='Cluster Agglo', right_on='Regularity')

clean_d = clean_d[clean_d.Segment != 'Best']
clean_d = clean_d[clean_d.Segment != 'One&Done']
clean_d = clean_d[clean_d.Segment != 'New']

#metrics = clean_d.filter(items=['Lifespan','AYMGSForcast', 'Regularity', 'PreferenceConfidence', 'Cluster KMeans'], axis=1)
metrics = nd1
metrics.replace([np.inf, -np.inf], np.nan, inplace=True)

metrics.fillna(0, inplace=True)

Lifespan_ = metrics.filter(items = ['Lifespan'])
Lifespan_mean = Lifespan_['Lifespan'].mean(skipna=True)
Lifespan_median = Lifespan_.median(axis=0)['Lifespan']
#print('Lifespan mean ', Lifespan_mean)

AYMGSForcast_ = metrics.filter(items = ['AYMGSForcast'])
AYMGSForcast_mean = AYMGSForcast_['AYMGSForcast'].mean(skipna=True)
AYMGSForcast_median = AYMGSForcast_.median(axis=0)['AYMGSForcast']
#print('AYMGSForcast_mean mean ', AYMGSForcast_mean)

Regularity_ = metrics.filter(items= ['Regularity'])
Regularity_mean = Regularity_['Regularity'].mean(skipna=True)
Regularity_median = Regularity_.median(axis=0)['Regularity']
#print('Regularity_mean mean ', Regularity_mean)

PrefCon_ = metrics.filter(items= ['PreferenceConfidence'])
PrefCon_mean = PrefCon_['PreferenceConfidence'].mean(skipna=True)
PrefCon_median = PrefCon_.median(axis=0)['PreferenceConfidence']
#print('PrefCon_mean mean ', PrefCon_mean)

EngagementConsistency_ = clean_d.filter(items=['EngagementConsistency'])
EngagementConsistency_mean = EngagementConsistency_.mean(axis=0)['EngagementConsistency']

Entropy_ = clean_d.filter(items= ['Entropy'])
Entropy_mean = Entropy_.mean(axis=0)['Entropy']


# Initialize a 2x2 grid of subplots
def log_transform(df):
    # Filter out zeros or negative values (log undefined)
    df_filtered = df[df > 0]
    # Apply log transformation
    return np.log10(df_filtered)


fig, axs = plt.subplots(2, 2, figsize=(10, 8))
Lifespan_log = log_transform(Lifespan_)
AYMGSForcast_log = log_transform(AYMGSForcast_)
Regularity_log = log_transform(Regularity_)
PrefCon_log = log_transform(PrefCon_)
'''
# Plot histograms for each column
axs[0, 0].hist(Lifespan_log['Lifespan'], bins=7, color='blue', alpha=0.7)
axs[0, 0].set_title('Distribution of Lifespan')

axs[0, 1].hist(AYMGSForcast_log['AYMGSForcast'], bins=10, color='green', alpha=0.7)
axs[0, 1].set_title('Distribution of PAYMScore')

axs[1, 0].hist(Regularity_log['Regularity'], bins=7, color='red', alpha=0.7)
axs[1, 0].set_title('Distribution of Regularity')

axs[1, 1].hist(PrefCon_log['PreferenceConfidence'], bins=6, color='purple', alpha=0.7)
axs[1, 1].set_title('Distribution of PreferenceConfidence')

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plot
#plt.show()
'''
def create_binary(column, threshold):
    return (column > threshold).astype(int)


sns.heatmap(metrics, annot=True, cmap="Blues")
#plt.show()

#################################### Bayesian Network ########################################
metrics = metrics[metrics['Lifespan'] > 0] 
metrics = metrics[metrics['Regularity'] > 0]
metrics = metrics[metrics['PreferenceConfidence'] > 0] 
#print(metrics[['Lifespan', 'Regularity', 'AYMGSForcast', 'PreferenceConfidence']].describe())

metrics['Lifespan_Bin'] = create_binary(metrics['Lifespan'], Lifespan_mean)
metrics['Regularity_Bin'] = create_binary(metrics['Regularity'], Regularity_mean)
metrics['AYMGSForecast_Bin'] = create_binary(metrics['AYMGSForcast'], AYMGSForcast_mean)
metrics['PreferenceConfidence_Bin'] = create_binary(metrics['PreferenceConfidence'], PrefCon_mean)
metrics['ClusterKMeans_Bin'] = metrics['Cluster KMeans']


# Keep only binned columns
metrics_discrete = metrics[['Lifespan_Bin', 'Regularity_Bin', 'AYMGSForecast_Bin', 'PreferenceConfidence_Bin', 'ClusterKMeans_Bin']]
#print(metrics_discrete[['Lifespan_Bin', 'Regularity_Bin', 'AYMGSForecast_Bin', 'PreferenceConfidence_Bin']].describe())

metrics_discrete = metrics_discrete.astype(int)
metrics_discrete = metrics_discrete[metrics_discrete["ClusterKMeans_Bin"] != 0]

cpd_regularity = metrics_discrete.groupby(['Lifespan_Bin', 'PreferenceConfidence_Bin'])['Regularity_Bin'].value_counts(normalize=True).unstack().fillna(0)
#cpd_cluster = metrics_discrete.groupby(['Lifespan_Bin', 'Regularity_Bin', 'AYMGSForecast_Bin', 'PreferenceConfidence_Bin'])['ClusterKMeans_Bin'].value_counts(normalize=True).unstack().fillna(0)

# Ensure ClusterKMeans_Bin contains only 1,2,3,4
metrics_discrete = metrics_discrete[metrics_discrete["ClusterKMeans_Bin"].isin([1, 2, 3, 4])]

# Step 1: Identify missing parent state combinations
expected_combinations = list(itertools.product([0, 1], repeat=4))  # 16 combinations

actual_combinations = set(
    [tuple(x) for x in metrics_discrete[['Lifespan_Bin', 'Regularity_Bin', 'AYMGSForecast_Bin', 'PreferenceConfidence_Bin']]
     .drop_duplicates()
     .to_numpy()]
)

missing_combinations = set(expected_combinations) - actual_combinations
print("Missing parent state combinations:", missing_combinations)

# Step 2: Create a DataFrame for missing combinations
missing_rows = pd.DataFrame(missing_combinations, columns=['Lifespan_Bin', 'Regularity_Bin', 'AYMGSForecast_Bin', 'PreferenceConfidence_Bin'])
missing_rows["ClusterKMeans_Bin"] = 1  # Assign a default cluster (adjust as needed)

# Append missing rows using pd.concat()
metrics_discrete = pd.concat([metrics_discrete, missing_rows], ignore_index=True)

# Step 3: Recompute cpd_cluster
cpd_cluster = pd.crosstab(
    index=[
        metrics_discrete['Lifespan_Bin'], 
        metrics_discrete['Regularity_Bin'], 
        metrics_discrete['AYMGSForecast_Bin'], 
        metrics_discrete['PreferenceConfidence_Bin']
    ],
    columns=metrics_discrete['ClusterKMeans_Bin'],
    normalize='index'
).fillna(0)

# Ensure all 16 parent state combinations exist
cpd_cluster = cpd_cluster.reindex(index=expected_combinations, fill_value=0)

# Step 4: Verify the shape
print("Final cpd_cluster shape:", cpd_cluster.shape)  # Should be (16,4)
if cpd_cluster.shape != (16,4):
    print("Error: Shape is still incorrect. Check missing values.")



print(metrics_discrete['ClusterKMeans_Bin'].value_counts()) 
print(cpd_cluster.info())


#Takes average of these to generate probabilites of success and failure ie High/Low
P_high_LP = metrics['Lifespan_Bin'].mean()
P_low_LP = 1 - P_high_LP

P_high_PC= metrics['PreferenceConfidence_Bin'].mean()
P_low_PC = 1 - P_high_PC

P_high_AF = metrics['AYMGSForecast_Bin'].mean()
P_low_AF = 1 - P_high_AF


#Converts these probabilities to something that can be used in the Bayesian Network

cpd_lifespan = TabularCPD(
    variable='Lifespan_Bin',
    variable_card=2,  
    values=[[P_high_LP], [P_low_LP]]
)
cpd_prefcon = TabularCPD(
    variable='PreferenceConfidence_Bin',
    variable_card=2,  
    values=[[P_high_PC],[P_low_PC] ]
)
cpd_AYGMSForecast = TabularCPD(
     variable='AYMGSForecast_Bin',
    variable_card=2,  
    values=[[P_high_AF], [P_low_AF]]
)


cpd_regularity = TabularCPD(
    variable='Regularity_Bin',
    variable_card=2,  
    values=cpd_regularity.values.T,  
    evidence=['PreferenceConfidence_Bin', 'Lifespan_Bin'],
    evidence_card=[2, 2]
)

cpd_cluster = TabularCPD(
    variable='ClusterKMeans_Bin',
    variable_card=4, 
    values=cpd_cluster.values.T,
    evidence=['Lifespan_Bin', 'Regularity_Bin', 'AYMGSForecast_Bin', 'PreferenceConfidence_Bin'],
    evidence_card=[2, 2, 2, 2]
)

model = BayesianNetwork([
    ("Lifespan_Bin", "Regularity_Bin"),
    ("PreferenceConfidence_Bin", "Regularity_Bin"),
    ("Lifespan_Bin", "ClusterKMeans_Bin"),
    ("AYMGSForecast_Bin", "ClusterKMeans_Bin"),
    ("Regularity_Bin", "ClusterKMeans_Bin"),
    ("PreferenceConfidence_Bin", "ClusterKMeans_Bin")
])

model.add_cpds(cpd_lifespan, cpd_prefcon, cpd_AYGMSForecast, cpd_regularity, cpd_cluster)
print(model.check_model())  # Debugging: This will tell you if there's an issue with CPDs
inference = VariableElimination(model)

combinations = list(itertools.product([0, 1], repeat=4))  # 4 binary variables

# Initialize the Bayesian Network and inference object (assuming the model is already defined)
inference = VariableElimination(model)
'''
# Iterate through all combinations of values for the parent nodes
for combination in combinations:
    evidence = {
        'Lifespan_Bin': combination[0],
        'Regularity_Bin': combination[1],
        'AYMGSForecast_Bin': combination[2],
        'PreferenceConfidence_Bin': combination[3]
    }
    
    # Perform inference for ClusterKMeans_Bin given the current evidence
    query = inference.query(variables=['ClusterKMeans_Bin'], evidence=evidence)
    
    # Print the results
    print(f"Evidence: {evidence}")
    print(query)
    print('-' * 50)  # Separator between each query result

query1 = inference.query(variables=['ClusterKMeans_Bin'], 
                        evidence={'Lifespan_Bin': 1,})
query2 = inference.query(variables=['ClusterKMeans_Bin'], 
                        evidence={'Regularity_Bin': 1,})
query3 = inference.query(variables=['ClusterKMeans_Bin'], 
                        evidence={'AYMGSForecast_Bin': 1,})
query4 = inference.query(variables=['ClusterKMeans_Bin'], 
                        evidence={'PreferenceConfidence_Bin': 1,})
 # Print the results
print('Lifespan_Bin')
print(query1)
print('Regularity_Bin')
print(query2)
print('AYMGSForecast_Bin')
print(query3)
print('PreferenceConfidence_Bin')
print(query4)
                        '''


query5 = inference.query(variables=['ClusterKMeans_Bin'], 
                        evidence={'Lifespan_Bin': 0,'Regularity_Bin': 0,
                                  'AYMGSForecast_Bin': 0,'PreferenceConfidence_Bin': 0})
print('All zeros ',query5)
