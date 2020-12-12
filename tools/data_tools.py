import pandas as pd
from sklearn.preprocessing import StandardScaler
import tools.general_tools as g_tools
from tools.general_tools import Obj
import numpy as np


def remove_features(data_frame: pd.DataFrame, features_to_remove: list) -> pd.DataFrame:
    features_keep = []
    for feature in data_frame.columns:
        matches = []
        for feature_remove in features_to_remove:
            if feature_remove not in feature:
                matches.append(False)
            else:
                matches.append(True)
                break

        if not any(matches):
            features_keep.append(feature)

    data_frame = data_frame[features_keep]
    return data_frame


def keep_features(data_frame: pd.DataFrame, features_to_keep: list) -> pd.DataFrame:
    features_keep = []
    for feature in data_frame.columns:
        matches = []
        for feature_keep in features_to_keep:
            if feature_keep in feature:
                matches.append(True)
                break
            else:
                matches.append(False)

        if any(matches):
            features_keep.append(feature)

    data_frame = data_frame[features_keep]
    return data_frame


def center_unit_variance(data, return_dataframe=False, col_names=None):
    # Centere training data to zero with a unit variance
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy(copy=True)

    if return_dataframe:
        data = StandardScaler().fit_transform(data)
        return pd.DataFrame(data, columns=col_names)
    else:
        return StandardScaler().fit_transform(data)


def preprocess(src, feature_scale=False, return_data=False, save=False, file_name=None):

    raw_data = pd.read_csv(src)
    features_meta_data = [
        "Metadata", "ImageNumber", "ObjectNumber", "AreaShape_EulerNumber",
        "Children_OverlapRegions_Count", "Number_Object_Number", "Parent_Nuclei",
        "Parent_V2R_cells"
    ]
    meta_data = keep_features(raw_data, features_meta_data)
    pd_data = remove_features(raw_data, features_meta_data)
    pd_data = pd_data.astype(dtype="float64")

    # Remove invalid samples
    idx_list = []
    for idx, row in pd_data.iterrows():
        if pd.isna(row["Correlation_Manders_ER_otsu_V2R_otsu"]) or pd.isna(row["Correlation_Manders_V2R_otsu_ER_otsu"]):
            idx_list.append(idx)
    pd_data = pd_data.drop(index=idx_list).copy(deep=True)
    meta_data = meta_data.drop(index=idx_list).copy(deep=True)

    # parse mutant IDs
    mutants = []
    for idx in meta_data.index:
        file_loc = meta_data.Metadata_FileLocation.loc[idx]

        if file_loc[-8: -4] == 'con_':
            mutants.append(0)
        else:
            mutant_num = int(file_loc[-7: -4])
            mutants.append(mutant_num)

    mutants = np.asarray(mutants)
    mutant_ids, mutant_counts = np.unique(mutants, return_counts=True)
    meta_data.insert(0, 'mutant id', mutants, allow_duplicates=False)

    # Impute training data NaNs with 0
    pd_data.fillna(float(0), inplace=True)

    if feature_scale:
        pd_data = center_unit_variance(pd_data, return_dataframe=True, col_names=pd_data.columns)

    if save:
        g_tools.local_data_save(data=Obj(
            meta_data=meta_data,
            training_data=pd_data
        ), file_name=file_name)

    if return_data:
        return Obj(
            meta_data=meta_data,
            training_data=pd_data
        )
