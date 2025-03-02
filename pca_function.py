import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

classifier = LogisticRegression(random_state = 0)
sc = StandardScaler()

raw_d = pd.read_csv('anon_PatronDetails - extended.csv')
#raw_d.info()

cut_d = raw_d.filter(items=['ClassicalScore', 'ChoralScore', 'ContemporaryScore', 'DanceScore','BrassScore',
                       'Recency','Frequency','Monetary','Lifespan','GrowthScore', 'AYM', 'RFMScore','CLV_Score'], axis=1)
#cut_d.info()

## Excludes scores in PCA and heatmap for the future
values = cut_d.iloc[:,0:11]
values.info()
#values_1 = cut_d.iloc[:,0:10]
y_rfm = cut_d.iloc[:, 11]
y_clv = cut_d.iloc[:, 12]
y_rfm.info()
y_clv.info()

scaler = StandardScaler()

X_scale = scaler.fit_transform(values)
X_scale_df = pd.DataFrame(X_scale, columns=values.columns)
X = X_scale_df.values
#print(X)
'''
X1_scale = scaler.fit_transform(values_1)
X1_scale_df = pd.DataFrame(X1_scale, columns=values_1.columns)
X1 = X1_scale_df.values

'''

kmeans=KMeans(n_clusters=4)


## heatmap data
data_heatmap= scaler.fit_transform(cut_d)
scaled_data_df = pd.DataFrame(data_heatmap, columns=cut_d.columns)
#data_rescaled_df = pd.DataFrame(X, columns=X.columns)
data = scaled_data_df.corr()
sns.heatmap(data, annot=True, cmap="crest")
plt.show()

pca=PCA(n_components=4)

reduced_X=pd.DataFrame(data=pca.fit_transform(X),columns=['PC1','PC2','PC3','PC4',])
k_means_pca = kmeans.fit(reduced_X)
#centers=pca.transform(kmeans.cluster_centers_)

## Elbow Plot for PCA
wcss = {} 
for i in range(1, 9): 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(reduced_X) 
    wcss[i] = kmeans.inertia_
     
plt.plot(wcss.keys(), wcss.values(), 'gs-')
plt.xlabel("Values of 'k'")
plt.ylabel('WCSS')
#plt.show()

## 3D ScatterPlot PCA Components =3 
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(reduced_X['PC1'],reduced_X['PC2'], reduced_X['PC3'],c=kmeans.labels_)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Patron Cluster')
ax.get_tightbbox()
#plt.show()

#loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2','PC3'], index=values.columns)
#print(loadings)

component_df=pd.DataFrame(pca.components_,index=reduced_X.index,columns=values.columns)
sns.heatmap(component_df, annot=True, cmap="crest")
plt.show()

'''

pca_dbscan=PCA(n_components=2)

reduced_X2=pd.DataFrame(data=pca_dbscan.fit_transform(X1),columns=['PCA1','PCA2'])


db = DBSCAN(eps=0.4, min_samples=10).fit(reduced_X2)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
 
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
 
# Plot result
 
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = ['y', 'b', 'g', 'r',]
print(colors)
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
 
    class_member_mask = (labels == k)
 
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k',
             markersize=6)
 
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k',
             markersize=6)
 
plt.title('number of clusters: %d' % n_clusters_)
plt.show()

'''




