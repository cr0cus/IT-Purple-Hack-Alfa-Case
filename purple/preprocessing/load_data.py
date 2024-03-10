import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def extract_features_labels(data_df: pd.DataFrame, categorical_labels: bool = True) -> tuple[np.ndarray, np.ndarray]:
    prepared_data_df = data_df.copy()
    cat_cols = [
        "okved", "segment", "start_cluster", "ogrn_month", "ogrn_year",
    ]

    drop_cat_cols = ["channel_code", "city", "city_type", "index_city_code"]

    x = prepared_data_df.drop(["id", "date", "end_cluster"], axis=1)
    x = x.drop(drop_cat_cols, axis=1)
    x = pd.get_dummies(x, columns=cat_cols).to_numpy()
    y = prepared_data_df["end_cluster"].to_numpy()

    if categorical_labels:
        categories = sorted(list(set(y)))
        categories_2d = np.expand_dims(categories, axis=1)
        ohe = OneHotEncoder(sparse_output=False).fit(categories_2d)
        y = ohe.transform(np.expand_dims(y, axis=1))

    return x, y


def extract_subj_features_labels(data_df: pd.DataFrame, categorical_labels: bool = True) -> tuple[np.ndarray, np.ndarray]:
    prepared_data_df = data_df.copy()
    cat_cols = [
        "okved", "segment", "start_cluster", "ogrn_month", "ogrn_year",
    ]

    drop_cat_cols = ["channel_code", "city", "city_type", "index_city_code"]

    x = prepared_data_df.drop(["id", "date", "end_cluster"], axis=1)
    x = x.drop(drop_cat_cols, axis=1)
    x = pd.get_dummies(x, columns=cat_cols).to_numpy()
    y = prepared_data_df["end_cluster"].to_numpy()

    if categorical_labels:
        categories = sorted(list(set(y)))
        categories_2d = np.expand_dims(categories, axis=1)
        ohe = OneHotEncoder(sparse_output=False).fit(categories_2d)
        y = ohe.transform(np.expand_dims(y, axis=1))

    x = x.reshape((x.shape[0] // 3, 3, x.shape[1]))
    y = y[::3]
    
    return x, y


if __name__ == "__main__":
    data_path = "../../data/train_data.pqt"

    train_df = pd.read_parquet(data_path)

    x, y = extract_features_labels(train_df)
    print(x.shape)
    print(y.shape)
    print(y[0])

    x, y = extract_subj_features_labels(train_df)
    print(x.shape)
    print(y.shape)
