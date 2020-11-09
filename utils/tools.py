import pandas as pd


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
