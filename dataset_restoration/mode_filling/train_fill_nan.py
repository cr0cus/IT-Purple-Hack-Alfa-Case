import numpy as np
import pandas as pd

data_path = "../../data/train_data.pqt"
train_df = pd.read_parquet(data_path)

column_names = train_df.columns.to_list()

pd.options.display.max_rows = 100
print(train_df.isna().sum())

for column in column_names:
    train_df.loc[train_df[column].isnull(), column] = train_df[column].mode()[0]

print(train_df.isna().sum())
train_df.to_parquet('data/train_data_modefilled.pqt')


