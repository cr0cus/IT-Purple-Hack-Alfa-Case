import pandas as pd
import numpy as np

pd.options.display.max_rows = 100

df = pd.read_parquet("../data/train_data.pqt")

required_cols = ['balance_amt_avg', 'balance_amt_max', 'balance_amt_min',
       'balance_amt_day_avg', 'sum_of_paym_2m', 'sum_of_paym_6m', 'sum_of_paym_1y', 'sum_a_oper_1m',
       'cnt_a_oper_1m', 'sum_b_oper_1m', 'cnt_b_oper_1m', 'sum_c_oper_1m',
       'cnt_c_oper_1m', 'sum_deb_d_oper_1m', 'cnt_deb_d_oper_1m',
       'sum_cred_d_oper_1m', 'cnt_cred_d_oper_1m', 'sum_deb_e_oper_1m',
       'cnt_deb_e_oper_1m', 'cnt_days_deb_e_oper_1m', 'sum_cred_e_oper_1m',
       'cnt_cred_e_oper_1m', 'cnt_days_cred_e_oper_1m', 'sum_deb_f_oper_1m',
       'cnt_deb_f_oper_1m', 'cnt_days_deb_f_oper_1m', 'sum_cred_f_oper_1m',
       'cnt_cred_f_oper_1m', 'cnt_days_cred_f_oper_1m', 'sum_deb_g_oper_1m',
       'cnt_deb_g_oper_1m', 'cnt_days_deb_g_oper_1m', 'sum_cred_g_oper_1m',
       'cnt_cred_g_oper_1m', 'cnt_days_cred_g_oper_1m', 'sum_deb_h_oper_1m',
       'cnt_deb_h_oper_1m', 'cnt_days_deb_h_oper_1m', 'sum_cred_h_oper_1m',
       'cnt_cred_h_oper_1m', 'cnt_days_cred_h_oper_1m', 'sum_a_oper_3m',
       'cnt_a_oper_3m', 'sum_b_oper_3m', 'cnt_b_oper_3m', 'sum_c_oper_3m',
       'cnt_c_oper_3m', 'sum_deb_d_oper_3m', 'cnt_deb_d_oper_3m',
       'sum_cred_d_oper_3m', 'cnt_cred_d_oper_3m', 'sum_deb_e_oper_3m',
       'cnt_deb_e_oper_3m', 'cnt_days_deb_e_oper_3m', 'sum_cred_e_oper_3m',
       'cnt_cred_e_oper_3m', 'cnt_days_cred_e_oper_3m', 'sum_deb_f_oper_3m',
       'cnt_deb_f_oper_3m', 'cnt_days_deb_f_oper_3m', 'sum_cred_f_oper_3m',
       'cnt_cred_f_oper_3m', 'cnt_days_cred_f_oper_3m', 'sum_deb_g_oper_3m',
       'cnt_deb_g_oper_3m', 'cnt_days_deb_g_oper_3m', 'sum_cred_g_oper_3m',
       'cnt_cred_g_oper_3m', 'cnt_days_cred_g_oper_3m', 'sum_deb_h_oper_3m',
       'cnt_deb_h_oper_3m', 'cnt_days_deb_h_oper_3m', 'sum_cred_h_oper_3m',
       'cnt_cred_h_oper_3m', 'cnt_days_cred_h_oper_3m']

subjs = list(df["id"].unique())
nan_cols = set()
cnt_nan_subjs = 0
for subj in subjs[:100]:
    f = False
    subj_df = df[df["id"] == subj]
    subj_df = subj_df[required_cols]
    sum_nan = subj_df.isna().sum()
    for i in range(len(sum_nan)):
        if sum_nan[i] == 3:
            nan_cols.add(required_cols[i])
            f = True
    if f:
        cnt_nan_subjs += 1
print(cnt_nan_subjs)
print(nan_cols)

for i in range(len(required_cols)):
    if required_cols[i] not in nan_cols:
        print(required_cols[i])