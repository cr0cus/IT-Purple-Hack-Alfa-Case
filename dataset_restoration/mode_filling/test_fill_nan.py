import pandas as pd

pd.options.display.max_rows = 100

train_df = pd.read_parquet("../../data/train_data_modefilled.pqt")
test_df = pd.read_parquet("../../data/test_data.pqt")

column_names = test_df.columns.to_list()

test_df.loc[test_df["date"] == "month_6", "start_cluster"] = test_df[test_df["date"] == "month_5"]["start_cluster"].values
test_df.loc[test_df["segment"].isnull(), "segment"] = train_df["segment"].mode()[0]

column_names.remove("start_cluster")

for column in column_names:
    test_df.loc[test_df[column].isnull(), column] = train_df[column].mode()[0]

print(test_df.isna().sum())

test_df.to_parquet('data/test_data_modefilled.pqt')
