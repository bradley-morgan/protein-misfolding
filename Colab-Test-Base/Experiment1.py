'''
Experiment 1: Apply 1D clustering to just the manders score feature
'''

import pandas as pd
import helpers.tools as tools
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import jenkspy

raw_data = pd.read_csv("../Data/raw/Ouput_coloc_pipline_vs15_Coloc-III_V2R_cyto.csv")

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

#Get data as numpy array
manders_data = pd_data["Correlation_Manders_ER_otsu_V2R_otsu"].values
# manders_data = manders_data.reshape(1, -1)

# Z Mean Normalization on data
z_mean_scaler = StandardScaler()
# manders_data = z_mean_scaler.fit_transform(manders_data)

# Order the data compute the natrual breaks in the data
manders_data_ordered = np.flip(np.sort(manders_data, axis=0))
breaks = jenkspy.jenks_breaks(manders_data_ordered, nb_class=3)

# Visualise distribution of each value
sns.set()
plt.figure(0)
y = [i for i in range(len(manders_data))]
plt.scatter(x=manders_data, y=y, marker='.')
plt.vlines(breaks,0, len(y), colors='black')
# plt.yticks([])

plt.figure(1)
sns.distplot(manders_data, hist=True, kde=True)
plt.vlines(breaks,0, 4, colors='black')
plt.show()


# pd_data["Correlation_Manders_ER_otsu_V2R_otsu"].plot.density()
# plt.show()
