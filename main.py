import pandas as pd
import utls.tools as tools
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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

#Get data as numpy array
np_data = pd_data.loc[:, pd_data.columns].values
manders_data = pd_data["Correlation_Manders_ER_otsu_V2R_otsu"].values
manders_data = manders_data.reshape(-1, 1)

# Z Mean Normalization on data
z_mean_scaler = StandardScaler()
np_data = z_mean_scaler.fit_transform(np_data)
# manders_data = z_mean_scaler.fit_transform(manders_data)

plt.figure(0)
plt.scatter(x=manders_data, y=[i for i in range(len(manders_data))], marker='.')
plt.yticks([])
plt.show()










