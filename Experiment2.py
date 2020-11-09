'''
Experiment 2: Apply 1D clustering to just the manders score feature
'''

import pandas as pd
import utls.tools as tools
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


raw_data = pd.read_csv("./Data/Ouput_coloc_pipline_vs15_Coloc-III_V2R_cyto.csv")

# Remove MetaData and features I dont know how to handle yet
features_meta_data = [
    "Metadata", "ImageNumber", "ObjectNumber", "AreaShape_EulerNumber",
    "Children_OverlapRegions_Count", "Number_Object_Number", "Parent_Nuclei",
    "Parent_V2R_cells"
]
meta_data = tools.keep_features(raw_data, features_meta_data)
pd_data = tools.remove_features(raw_data, features_meta_data)
pd_data = pd_data.astype(dtype="float64")

# Remove and Fill in nans
pd_data = pd_data.fillna(float(0))
z_mean_scaler = StandardScaler()
scaled_data = pd.DataFrame(z_mean_scaler.fit_transform(pd_data))

# Find elbow for best number of clusters
sum_of_sqrd_distances = []
K = 21
with tqdm(total=K, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}') as progress_bar:
    for k in range(1, K):
        km = KMeans(n_clusters=k)
        km = km.fit(scaled_data)
        sum_of_sqrd_distances.append(km.inertia_)
        progress_bar.update(1)

sns.set()
plt.figure(0)
plt.plot(range(1, K), sum_of_sqrd_distances, '.-')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.xticks(range(1, K))
plt.ylabel('Sum of Squared Distances')
plt.show()

# Perform K means with optim K
km = KMeans(n_clusters=8)
km = km.fit(scaled_data)


