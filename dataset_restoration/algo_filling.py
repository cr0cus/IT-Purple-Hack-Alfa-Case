import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score

pd.options.display.max_rows = 100

df = pd.read_parquet("../data/train_data.pqt")
to_delete = ['id', 'date', 'end_cluster',
             'channel_code',
             'city', 'city_type', 'index_city_code', 'ogrn_days_end_month',
             'ogrn_days_end_quarter', 'ogrn_month', 'ogrn_year', 'ft_registration_date', 'max_founderpres', 'min_founderpres', 'ogrn_exist_months',
             'okved',
             # 'start_cluster', 'segment'
             ]
to_cat = ['start_cluster', 'segment']
df = pd.get_dummies(df, columns=to_cat)
df = df.drop(columns=to_delete)
print(df.isna().sum())
df = df.dropna()
# print(train_df.shape)
to_restore = ["balance_amt_avg", "balance_amt_max", "balance_amt_min", "balance_amt_day_avg", "sum_of_paym_2m", "sum_of_paym_6m", "sum_of_paym_1y",
              "cnt_a_oper_1m", "cnt_b_oper_1m", "cnt_c_oper_1m", "cnt_deb_d_oper_1m", "cnt_cred_d_oper_1m", "cnt_deb_e_oper_1m", "cnt_days_deb_e_oper_1m",
              "cnt_cred_e_oper_1m", "cnt_days_cred_e_oper_1m", "cnt_deb_f_oper_1m", "cnt_days_deb_f_oper_1m", "cnt_cred_f_oper_1m", "cnt_days_cred_f_oper_1m",
              "cnt_deb_g_oper_1m", "cnt_days_deb_g_oper_1m", "cnt_cred_g_oper_1m", "cnt_days_cred_g_oper_1m", "cnt_deb_h_oper_1m", "cnt_days_deb_h_oper_1m",
              "cnt_cred_h_oper_1m", "cnt_days_cred_h_oper_1m", "cnt_a_oper_3m", "cnt_b_oper_3m", "cnt_c_oper_3m", "cnt_deb_d_oper_3m", "cnt_cred_d_oper_3m",
              "cnt_deb_e_oper_3m", "cnt_days_deb_e_oper_3m", "cnt_cred_e_oper_3m", "cnt_days_cred_e_oper_3m", "cnt_deb_f_oper_3m", "cnt_days_deb_f_oper_3m",
              "cnt_cred_f_oper_3m", "cnt_days_cred_f_oper_3m", "cnt_deb_g_oper_3m", "cnt_days_deb_g_oper_3m", "cnt_cred_g_oper_3m", "cnt_days_cred_g_oper_3m",
              "cnt_deb_h_oper_3m", "cnt_days_deb_h_oper_3m", "cnt_cred_h_oper_3m", "cnt_days_cred_h_oper_3m"]

train_df = df.drop(columns=to_restore)

results_r2 = []
for idx, column in enumerate(to_restore):
    X_train, X_test, y_train, y_test = train_test_split(train_df, df[column], test_size=0.2)

    # model = KNeighborsRegressor(n_neighbors=3)
    # model = LinearRegression()
    # model = SGDRegressor()
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #result_mse = mean_squared_error(y_test, y_pred)
    result_r2 = r2_score(y_test, y_pred)

    #results_mse.append(result_mse)
    results_r2.append(result_r2)

    print(column)
    #print(f"mse: {result_mse}")
    print(f"r2: {result_r2}")
    print(f'{(idx+1) * 100 // len(to_restore)}%\n')

print("---------------------------------")
#print(f"avg mse {np.mean(results_mse)}")
#print(f"median mse {np.median(results_mse)}")
print(f"avg r2 {np.mean(results_r2)}")
print(f"median r2 {np.median(results_r2)}")