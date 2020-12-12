

import pandas as pd
import helpers.tools as tools
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from pca import pca
from tqdm import tqdm
import jenkspy
from sklearn.metrics import silhouette_score

sns.set()
data_path = "../Data/raw/Ouput_coloc_pipline_vs14b_Coloc-IV2R_cyto.csv"
raw_data = pd.read_csv(data_path)

features_meta_data = [
    "Metadata", "ImageNumber", "ObjectNumber", "AreaShape_EulerNumber",
    "Children_OverlapRegions_Count", "Number_Object_Number", "Parent_Nuclei",
    "Parent_V2R_cells"
]
meta_data = tools.keep_features(raw_data, features_meta_data)
pd_data = tools.remove_features(raw_data, features_meta_data)
pd_data = pd_data.astype(dtype="float64")

idx_list = []
for idx, row in pd_data.iterrows():
    if pd.isna(row["Correlation_Manders_ER_otsu_V2R_otsu"]) or pd.isna(row["Correlation_Manders_V2R_otsu_ER_otsu"]):
        idx_list.append(idx)
pd_data = pd_data.drop(index=idx_list).copy(deep=True)
meta_data = meta_data.drop(index=idx_list).copy(deep=True)

nans = pd_data["Correlation_Manders_ER_otsu_V2R_otsu"].isna().value_counts()
nan = pd_data["Correlation_Manders_V2R_otsu_ER_otsu"].isna().value_counts()

# get mutants
mutants = []
for idx in meta_data.index:
    file_loc = meta_data.Metadata_FileLocation.loc[idx]

    if file_loc[-8 : -4] == 'con_':
        mutants.append(0)
    else:
        mutant_num = int(file_loc[-7 : -4])
        mutants.append(mutant_num)

mutants = np.asarray(mutants)
mutant_ids, mutant_counts = np.unique(mutants, return_counts=True)
meta_data.insert(0, 'mutant id', mutants, allow_duplicates=False)


# Fill NaNs & Create clean centered and unit variance dataset
pd_data.fillna(value=float(0), inplace=True)
pd_data_values = pd_data.to_numpy(copy=True)
clean_data = StandardScaler().fit_transform(pd_data)

# This seems to be an outlier sample
clean_data = np.delete(clean_data, 3316, axis=0)
clean_pd_data = pd.DataFrame(clean_data, columns=pd_data.columns)
meta_data = meta_data.drop(3316, axis=0)

manders_data = clean_pd_data["Correlation_Manders_V2R_otsu_ER_otsu"].values

# Reshape Data so sklearn doesnt complain
manders_data_resh = manders_data.reshape(-1, 1)

# Calculate K-Clusters
K = 3
km = KMeans(n_clusters=K)
cluster_labels = km.fit_predict(manders_data_resh)
centroids = km.cluster_centers_

manders_data_ordered = np.flip(np.sort(manders_data, axis=0))
breaks = jenkspy.jenks_breaks(manders_data_ordered, nb_class=3)

# TODO Plot the Clusters, denisty and jenks plots

y = [i for i in range(len(manders_data))]
plt.figure(2, figsize=(12,8), dpi= 100)
ax = sns.distplot(manders_data_resh, hist=True, kde=True,norm_hist=True, bins=30)
patches = ax.patches
for bar in patches:
    color = None
    if bar._x0 <= breaks[1]:
        color = (102/255,194/255,165/255, .6)
    elif  bar._x0 > breaks[1] and bar._x1 < breaks[2]:
        color = (179/255,179/255,179/255, .6)
    else:
        color = (166/255,216/255,84/255, .6)

    bar._facecolor = color

plt.scatter(x=manders_data_resh, y=np.zeros_like(manders_data_resh) + 2,
            c=cluster_labels, cmap='Set2')
plt.vlines(breaks, 0, 2.1, colors='black', linestyle='dashed')
plt.legend(['Kernal Density Estimator', '1D K-Means Clusters', '1D Jenkins Break Optimisation'])
plt.title('1D Cluster Analysis of Mutant Cells')
plt.xlabel('Manders Coefficient: Feature Standardised(mean=0, unit-variance)')
for idx, cent in enumerate(centroids):
    plt.annotate(f'X Centroid{idx}', (cent, 0))

ax_x = plt.gca()
leg = ax.get_legend()
leg.legendHandles[1].set_color((166/255,216/255,84/255, .6))

# TODO Compute Percentages for each centroid
mutants_per_cluster = {}
for i in range(K):
    mutants_per_cluster[f'cluster{i}'] = {'data': [], 'percentages': [], 'mutant_ids': []}

for idx, cluster_label in zip(meta_data.index, cluster_labels):
    mutant_id = meta_data.loc[idx]['mutant id']
    mutants_per_cluster[f'cluster{cluster_label}']['data'].append(mutant_id)

for key in mutants_per_cluster.keys():
    mutants_in_cluster = mutants_per_cluster[key]['data']
    m_ids, m_counts = np.unique(np.asarray(mutants_in_cluster), return_counts=True)

    m_ids = list(m_ids)
    m_counts = list(m_counts)
    ids_not_in_m_ids = []
    for idx, mutant_id in enumerate(mutant_ids):
        if mutant_id not in m_ids:
            ids_not_in_m_ids.append((mutant_id, idx))

    if len(ids_not_in_m_ids) != 0:
        for ids_not_in_m_id in ids_not_in_m_ids:
            m_ids.insert(ids_not_in_m_id[1], ids_not_in_m_id[0])
            m_counts.insert(ids_not_in_m_id[1], 0)

    for idx, m_id in enumerate(m_ids):
        m_count = m_counts[idx]
        mutant_id_pos = np.where(mutant_ids == m_id)[0][0]
        m_total = mutant_counts[mutant_id_pos]
        m_percentage = np.round((m_count / m_total) * 100, decimals=1)
        mutants_per_cluster[key]['percentages'].append(m_percentage)
        mutants_per_cluster[key]['mutant_ids'].append(m_id)



# Plot the Percentage of each mutant for each cluster
fig1, ax2 = plt.subplots(1, 1, figsize=(12,8), dpi= 100)
colors = ['#f28c91', '#8cf291', '#8c8cf7']
for key, color in zip(mutants_per_cluster.keys(), colors):
    ax2.plot(mutants_per_cluster[key]['percentages'],  marker='.', color=color)

ax2.set(xlabel='Mutant Ids', ylabel='Percentage of each Mutant in each Cluster')
ax2.legend(['Centroid 0', 'Centroid 1', 'Centroid 2'])
ax2.set_xticks(range(0, len(mutant_ids)))
# Set ticks labels for x-axis
ax2.set_xticklabels(mutant_ids)
ax2.grid(b=True)
plt.show()
