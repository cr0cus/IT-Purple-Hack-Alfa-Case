import purple.preprocessing.load_data as load_data
import purple.ml.fc as fc
#privet zaebal
import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.neural_network import MLPClassifier
import tensorflow as tf

print("Hello, Kitty!")

pd.options.display.max_rows = 100

train_df = pd.read_parquet("data/train_data_modefilled.pqt")
test_df = pd.read_parquet("data/test_data_modefilled.pqt")

last_m_test_df = test_df[test_df["date"] == "month_6"].reset_index(drop=True)

labels_train, x_train, y_train = load_data.extract_month_correlation_features_labels(train_df)
_, x_test, _ = load_data.extract_month_correlation_features_labels(test_df, train=False)

# le = LabelEncoder().fit(labels_train)
# y_train = le.transform(y_train)

y_train = load_data.one_hot_encoder(y_train, labels_train)

model = fc.FCClassifier(input_shape=x_train.shape[1])
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=15,
          batch_size=256)

test_pred_proba = model.predict(x_test)
# test_pred_proba = model.predict_proba(x_test)

test_pred_proba_df = pd.DataFrame(test_pred_proba, columns=labels_train)
sorted_classes = sorted(test_pred_proba_df.columns.to_list())
test_pred_proba_df = test_pred_proba_df[sorted_classes]

print(test_pred_proba_df.head(2))

sample_submission_df = pd.read_csv("data/sample_submission.csv")
sample_submission_df[sorted_classes] = test_pred_proba_df
sample_submission_df.to_csv("results/submission_fc_xgb_6.csv", index=False)







