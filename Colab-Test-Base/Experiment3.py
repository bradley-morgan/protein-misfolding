'''
Experiment 3: Apply Non-Linear Dimensionility Reduction
'''

import pandas as pd
import helpers.tools as tools
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tqdm import tqdm
from sklearn.metrics import silhouette_score


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

# Visualise with Isomap

model = Isomap(n_components=2, n_jobs=5)
projections = model.fit_transform(clean_data)

# plt.figure(0)
# plt.scatter(projections[:, 0], projections[:, 1],
#             c=clusters_labels.astype(np.float), edgecolor='k'
#             )
# plt.title('Isomap manifold')
# plt.xlabel(f'Dim 1')
# plt.ylabel(f'Dim 2')
# # for idx in range(len(projections)):
# #   plt.annotate(str(idx), (projections[idx, 0], projections[idx, 1]))
# plt.show()

# sil_scores = []
# K = 20
# with tqdm(total=K, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}') as progress_bar:
#     for k in range(2, K):
#         km = KMeans(n_clusters=k)
#         cluster_labels = km.fit_predict(projections)
#         sil = silhouette_score(projections, cluster_labels)
#         sil_scores.append(sil)
#         progress_bar.update(1)
#
# sns.set()
# plt.figure(0)
# plt.plot(range(2, K), sil_scores, '.-')
# plt.title('Silhouette Method for Optimal K')
# plt.xlabel('Number of Clusters')
# plt.xticks(range(1, K))
# plt.ylabel('Silhouette Score')
# plt.show()

# cluster data with all features
km = KMeans(n_clusters=4)
km = km.fit(projections)
clusters_labels = km.labels_

plt.figure(0)
plt.scatter(projections[:, 0], projections[:, 1],
            c=clusters_labels.astype(np.float), edgecolor='k'
            )
plt.title('Isomap manifold')
plt.xlabel(f'Dim 1')
plt.ylabel(f'Dim 2')
# for idx in range(len(projections)):
#   plt.annotate(str(idx), (projections[idx, 0], projections[idx, 1]))
plt.show()

