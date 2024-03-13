import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


def extract_features_labels(data_df: pd.DataFrame,
                            categorical_labels: bool = True,
                            train: bool = True) -> tuple[list, np.ndarray, np.ndarray]:
    prepared_data_df = data_df.copy()

    drop_cols = [
        'city', 'city_type', 'index_city_code', 'channel_code',
        'ogrn_days_end_month', 'ogrn_days_end_quarter', 'ogrn_month', 'ogrn_year',
        'ft_registration_date', 'max_founderpres', 'min_founderpres', 'ogrn_exist_months', 'okved',
    ]

    cat_cols = [
        "segment", "start_cluster"
    ]
    x = prepared_data_df.drop(["id", "date"], axis=1)
    if train:
        x = x.drop(["end_cluster"], axis=1)
    x = x.drop(drop_cols, axis=1)
    x = pd.get_dummies(x, columns=cat_cols).to_numpy()
    categories = None
    y = None
    if train:
        y = prepared_data_df["end_cluster"].to_numpy()

        categories = sorted(list(set(y)))

        if categorical_labels:
            categories_2d = np.expand_dims(categories, axis=1)
            ohe = OneHotEncoder(sparse_output=False).fit(categories_2d)
            y = ohe.transform(np.expand_dims(y, axis=1))

    return categories, x, y


def one_hot_encoder(y: np.ndarray, categories: list[str]) -> np.ndarray:
    categories_2d = np.expand_dims(categories, axis=1)
    ohe = OneHotEncoder(sparse_output=False).fit(categories_2d)
    y = ohe.transform(np.expand_dims(y, axis=1))

    return y


def extract_month_correlation_features_labels(data_df: pd.DataFrame, train: bool = True) -> tuple[list, np.ndarray, np.ndarray]:
    prepared_data_df = data_df.copy()

    drop_cols = [
        'city', 'city_type', 'index_city_code', 'channel_code',
        'ogrn_days_end_month', 'ogrn_days_end_quarter', 'ogrn_month', 'ogrn_year',
        'ft_registration_date', 'max_founderpres', 'min_founderpres', 'ogrn_exist_months', 'okved',
    ]

    cat_cols = [
        "segment", "start_cluster"
    ]

    x = prepared_data_df.drop(prepared_data_df[prepared_data_df.date == "month_4"].index)
    x = x.drop(["id", "date"], axis=1)
    if train:
        x = x.drop(["end_cluster"], axis=1)
    x = x.drop(drop_cols, axis=1)
    x = pd.get_dummies(x, columns=cat_cols).to_numpy()
    categories = None
    y = None
    if train:
        y = prepared_data_df["end_cluster"].to_numpy()

        categories = sorted(list(set(y)))

    if train:
        x = x.reshape((x.shape[0] // 3, 3, x.shape[1]))
    else:
        x = x.reshape((x.shape[0] // 2, 2, x.shape[1]))
    x_t = []
    for item in x:
        if len(item) == 3:
            x_t.append(item[1:])
        else:
            x_t.append(item)
    x_t = np.array(x_t)
    x_t = x_t.reshape((x_t.shape[0], x_t.shape[1] * x_t.shape[2]))
    if train:
        y = y[::3]

    return categories, x_t, y


if __name__ == "__main__":
    data_path = "../../data/test_data_modefilled.pqt"

    train_df = pd.read_parquet(data_path)

    category, x, y = extract_month_correlation_features_labels(train_df, train=False)
    print(x.shape)
    print(y.shape)
