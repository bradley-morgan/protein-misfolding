import tools.data_tools as d_tools
import tools.model_tools as m_tools
import tools.general_tools as g_tools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

file_name = 'vs14b-feature-scaled'
cluster_data = d_tools.preprocess(
    src='./Data/raw/Ouput_coloc_pipline_vs14b_Coloc-IV2R_cyto.csv',
    feature_scale=True,
    return_data=True,
)

data = d_tools.preprocess(
    src='./Data/raw/Ouput_coloc_pipline_vs14b_Coloc-IV2R_cyto.csv',
    feature_scale=False,
    return_data=True,
)

target_data = "Correlation_Manders_V2R_otsu_ER_otsu"
cluster_train = cluster_data.training_data[target_data].to_numpy()
cluster_train = cluster_train.reshape(-1, 1)

labels, centroids = m_tools.make_k_means_labels(K=2, data=cluster_train)
cluster_data.training_data.insert(0, "target", labels)
data.training_data.insert(0, "target", labels)
class_vals, class_counts = np.unique(labels, return_counts=True)
cluster_data(classes=class_vals, class_balance=class_counts)

g_tools.local_data_save(cluster_data, file_name+'-scaled')
g_tools.local_data_save(data, file_name+'-not-scaled')


