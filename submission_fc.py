import purple.preprocessing.load_data as load_data
import purple.ml.fc as fc

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import tensorflow as tf

pd.options.display.max_rows = 100

train_df = pd.read_parquet("data/train_data.pqt")
test_df = pd.read_parquet("data/test_data.pqt")
# print(test_df.head(3))
# print(test_df.shape)
# print(test_df.shape[0] // 3)
# print()
#
# print(test_df["date"].value_counts())
# print()

# print("month_5")
# print(test_df[['segment', 'start_cluster']][test_df["date"] == "month_5"].isna().sum())
#
# print("month_6")
# print(test_df[['segment', 'start_cluster']][test_df["date"] == "month_6"].isna().sum())

last_m_test_df = test_df[test_df["date"] == "month_6"].reset_index(drop=True)
last_m_test_df["start_cluster"] = test_df[test_df["date"] == "month_5"]["start_cluster"].values
last_m_test_df.loc[last_m_test_df["segment"].isnull(), "segment"] = test_df["segment"].mode()[0]

print("last month")
print(last_m_test_df[['segment', 'start_cluster']].isna().sum())

print(last_m_test_df.head(3))

labels_train, x_train, y_train = load_data.extract_features_labels(train_df, categorical_labels=False)
_, x_test, _ = load_data.extract_features_labels(last_m_test_df, categorical_labels=False, train=False)

y_train = load_data.one_hot_encoder(y_train, labels_train)

model = fc.FCClassifier(input_shape=x_train.shape[1])
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=10,
          batch_size=4096)

test_pred_proba = model.predict(x_test)

test_pred_proba_df = pd.DataFrame(test_pred_proba, columns=labels_train)
sorted_classes = sorted(test_pred_proba_df.columns.to_list())
test_pred_proba_df = test_pred_proba_df[sorted_classes]

print(test_pred_proba_df.head(2))

sample_submission_df = pd.read_csv("data/sample_submission.csv")
sample_submission_df[sorted_classes] = test_pred_proba_df
sample_submission_df.to_csv("results/baseline_submission_fc.csv", index=False)






