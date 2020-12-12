'''
Experiment 2: Apply 1D clustering to just the manders score feature
'''

import pandas as pd
import helpers.tools as tools
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tqdm import tqdm

data_path = '../Data/raw/Ouput_coloc_pipline_vs15_Coloc-III_V2R_cyto.csv'
sns.set()

raw_data = pd.read_csv(data_path)

features_meta_data = [
    "Metadata", "ImageNumber", "ObjectNumber", "AreaShape_EulerNumber",
    "Children_OverlapRegions_Count", "Number_Object_Number", "Parent_Nuclei",
    "Parent_V2R_cells"
]
meta_data = tools.keep_features(raw_data, features_meta_data)
pd_data = tools.remove_features(raw_data, features_meta_data)
pd_data = pd_data.astype(dtype="float64")

pd_data.fillna(value=float(0), inplace=True)
clean_data = StandardScaler().fit_transform(pd_data)
# This seems to be an outlier sample
clean_data = np.delete(clean_data, 3316, axis=0)

#

# PCA

pca = PCA()
pca.fit(clean_data)
pca_out = pca.transform(clean_data)
# Calculate % variance contribution of each component
percent_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
pca_labels = [f'PC{str(x)}' for x in range(1, len(percent_var)+1)]
pca_df = pd.DataFrame(pca_out, columns=pca_labels)

data_slice = percent_var[0:20]

plt.figure(0)
plt.bar(x=range(1, len(data_slice)+1), height=data_slice)
plt.ylabel("Percentage of Explained Variance")
plt.xlabel("Principal Components")
plt.title("PCA Scree Plot")
for idx, val in enumerate(data_slice):
    plt.text(idx + 0.8, val + 0.1, f'{str(val)}%')

# Plot pca Data
plt.figure(1)
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('Title Goes Here')
plt.xlabel(f'PC1 - {percent_var[0]}%')
plt.ylabel(f'PC2 - {percent_var[1]}%')
# for sample in pca_df.index:
#   plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))






#
# cluster_data = pca_df[['PC1', 'PC2']]
#


cluster_data = pca_df[['PC1', 'PC2']]

km = KMeans(n_clusters=9)
km = km.fit(cluster_data)
clusters_labels = km.labels_

fig = plt.figure()
# ax = Axes3D(fig)
plt.scatter(cluster_data.PC1.values, cluster_data.PC2.values,
               c=clusters_labels.astype(np.float), edgecolor='k')
plt.show()

# sil_scores = []
# K = 20
# with tqdm(total=K, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}') as progress_bar:
#     for k in range(2, K):
#         km = KMeans(n_clusters=k)
#         cluster_labels = km.fit_predict(cluster_data)
#         sil = silhouette_score(cluster_data, cluster_labels)
#         sil_scores.append(sil)
#         progress_bar.update(1)
#
# sns.set()
# plt.figure(2)
# plt.plot(range(2, K), sil_scores, '.-')
# plt.title('Silhouette Method for Optimal K')
# plt.xlabel('Number of Clusters')
# plt.xticks(range(1, K))
# plt.ylabel('Silhouette Score')
# plt.show()



