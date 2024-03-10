import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


def extract_features_labels(data_df: pd.DataFrame,
                            categorical_labels: bool = True,
                            train: bool = True) -> tuple[list, np.ndarray, np.ndarray]:
    prepared_data_df = data_df.copy()

    drop_cols = [
                 'balance_amt_avg', 'balance_amt_max', 'balance_amt_min', 'balance_amt_day_avg',
                 'channel_code', 'city', 'city_type', 'index_city_code',
                 'ogrn_days_end_month', 'ogrn_days_end_quarter', 'ogrn_month', 'ogrn_year',
                 'ft_registration_date', 'max_founderpres', 'min_founderpres', 'ogrn_exist_months', 'okved',
                 'sum_of_paym_2m', 'sum_of_paym_6m', 'sum_of_paym_1y',
                 'sum_a_oper_1m', 'cnt_a_oper_1m',
                 'sum_b_oper_1m', 'cnt_b_oper_1m',
                 'sum_c_oper_1m', 'cnt_c_oper_1m',
                 'sum_deb_d_oper_1m', 'cnt_deb_d_oper_1m',
                 'sum_cred_d_oper_1m', 'cnt_cred_d_oper_1m',
                 'sum_deb_e_oper_1m', 'cnt_deb_e_oper_1m', 'cnt_days_deb_e_oper_1m',
                 'sum_cred_e_oper_1m', 'cnt_cred_e_oper_1m', 'cnt_days_cred_e_oper_1m',
                 'sum_deb_f_oper_1m', 'cnt_deb_f_oper_1m', 'cnt_days_deb_f_oper_1m',
                 'sum_cred_f_oper_1m', 'cnt_cred_f_oper_1m', 'cnt_days_cred_f_oper_1m',
                 'sum_deb_g_oper_1m', 'cnt_deb_g_oper_1m', 'cnt_days_deb_g_oper_1m',
                 'sum_cred_g_oper_1m', 'cnt_cred_g_oper_1m', 'cnt_days_cred_g_oper_1m',
                 'sum_deb_h_oper_1m', 'cnt_deb_h_oper_1m', 'cnt_days_deb_h_oper_1m',
                 'sum_cred_h_oper_1m', 'cnt_cred_h_oper_1m', 'cnt_days_cred_h_oper_1m',
                 'sum_a_oper_3m', 'cnt_a_oper_3m',
                 'sum_b_oper_3m', 'cnt_b_oper_3m',
                 'sum_c_oper_3m', 'cnt_c_oper_3m',
                 'sum_deb_d_oper_3m', 'cnt_deb_d_oper_3m',
                 'sum_cred_d_oper_3m', 'cnt_cred_d_oper_3m',
                 'sum_deb_e_oper_3m', 'cnt_deb_e_oper_3m', 'cnt_days_deb_e_oper_3m',
                 'sum_cred_e_oper_3m', 'cnt_cred_e_oper_3m', 'cnt_days_cred_e_oper_3m',
                 'sum_deb_f_oper_3m', 'cnt_deb_f_oper_3m', 'cnt_days_deb_f_oper_3m',
                 'sum_cred_f_oper_3m', 'cnt_cred_f_oper_3m', 'cnt_days_cred_f_oper_3m',
                 'sum_deb_g_oper_3m', 'cnt_deb_g_oper_3m', 'cnt_days_deb_g_oper_3m',
                 'sum_cred_g_oper_3m', 'cnt_cred_g_oper_3m', 'cnt_days_cred_g_oper_3m',
                 'sum_deb_h_oper_3m', 'cnt_deb_h_oper_3m', 'cnt_days_deb_h_oper_3m',
                 'sum_cred_h_oper_3m', 'cnt_cred_h_oper_3m', 'cnt_days_cred_h_oper_3m'
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

    _, x, y = extract_features_labels(train_df)
    print(x.shape)
    print(y.shape)
    # print(y[0])
    #
    # x, y = extract_subj_features_labels(train_df)
    # print(x.shape)
    # print(y.shape)
