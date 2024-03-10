import numpy as np
import pandas as pd

data_path = "data/test_data.pqt"
train_df = pd.read_parquet(data_path)

column_name = train_df.columns.to_list()

print(len(column_name))
print(column_name)
# for column in column_name:
#     print(f"\n---{column}---\n")
#     print(train_df[column].describe())

