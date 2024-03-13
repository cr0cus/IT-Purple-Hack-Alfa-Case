import pandas as pd
import xgboost as xgb

pd.options.display.max_rows = 100

train_df = pd.read_parquet("../../data/train_data_xgbfilled.pqt")
test_df = pd.read_parquet("../../data/test_data.pqt")

test_df.loc[test_df["date"] == "month_6", "start_cluster"] = test_df[test_df["date"] == "month_5"]["start_cluster"].values
test_df.loc[test_df["segment"].isnull(), "segment"] = train_df["segment"].mode()[0]

to_delete = ['id', 'date', 'end_cluster', 'city', 'city_type', 'index_city_code', 'ogrn_days_end_month',
             'ogrn_days_end_quarter', 'ogrn_month', 'ogrn_year', 'ft_registration_date', 'max_founderpres', 'min_founderpres', 'ogrn_exist_months',
             'okved', 'channel_code']
to_delete_test = ['id', 'date', 'city', 'city_type', 'index_city_code', 'ogrn_days_end_month',
             'ogrn_days_end_quarter', 'ogrn_month', 'ogrn_year', 'ft_registration_date', 'max_founderpres', 'min_founderpres', 'ogrn_exist_months',
             'okved', 'channel_code']
to_restore = ["balance_amt_avg", "balance_amt_max", "balance_amt_min", "balance_amt_day_avg", "sum_of_paym_2m", "sum_of_paym_6m", "sum_of_paym_1y",
              "cnt_a_oper_1m", "cnt_b_oper_1m", "cnt_c_oper_1m", "cnt_deb_d_oper_1m", "cnt_cred_d_oper_1m", "cnt_deb_e_oper_1m", "cnt_days_deb_e_oper_1m",
              "cnt_cred_e_oper_1m", "cnt_days_cred_e_oper_1m", "cnt_deb_f_oper_1m", "cnt_days_deb_f_oper_1m", "cnt_cred_f_oper_1m", "cnt_days_cred_f_oper_1m",
              "cnt_deb_g_oper_1m", "cnt_days_deb_g_oper_1m", "cnt_cred_g_oper_1m", "cnt_days_cred_g_oper_1m", "cnt_deb_h_oper_1m", "cnt_days_deb_h_oper_1m",
              "cnt_cred_h_oper_1m", "cnt_days_cred_h_oper_1m", "cnt_a_oper_3m", "cnt_b_oper_3m", "cnt_c_oper_3m", "cnt_deb_d_oper_3m", "cnt_cred_d_oper_3m",
              "cnt_deb_e_oper_3m", "cnt_days_deb_e_oper_3m", "cnt_cred_e_oper_3m", "cnt_days_cred_e_oper_3m", "cnt_deb_f_oper_3m", "cnt_days_deb_f_oper_3m",
              "cnt_cred_f_oper_3m", "cnt_days_cred_f_oper_3m", "cnt_deb_g_oper_3m", "cnt_days_deb_g_oper_3m", "cnt_cred_g_oper_3m", "cnt_days_cred_g_oper_3m",
              "cnt_deb_h_oper_3m", "cnt_days_deb_h_oper_3m", "cnt_cred_h_oper_3m", "cnt_days_cred_h_oper_3m"]
to_cat = ['start_cluster', 'segment']

df_full = pd.read_parquet("../../data/train_data.pqt")
df_full = df_full.drop(columns=to_delete)
df_full = pd.get_dummies(df_full, columns=to_cat)
df_full = df_full.dropna()
df = df_full.drop(columns=to_restore)

for column in to_restore:
    x = df.to_numpy()
    y = df_full[column].to_numpy()

    model = xgb.XGBRegressor()
    model.fit(x, y)

    nan_df_features = test_df.copy()
    nan_df_features = pd.get_dummies(nan_df_features, columns=to_cat)
    nan_df_features = nan_df_features[nan_df_features[column].isnull()].copy()
    nan_df_features = nan_df_features.drop(columns=to_restore)
    nan_df_features = nan_df_features.drop(columns=to_delete_test).to_numpy()

    pred = model.predict(nan_df_features)
    test_df.loc[test_df[column].isnull(), column] = pred


print(test_df.isna().sum())
test_df.to_parquet('../../data/test_data_xgbfilled.pqt')
