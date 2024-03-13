import purple.preprocessing.load_data as load_data
import purple.ml.fc as fc

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb


def weighted_roc_auc(y_true, y_pred, labels, weights_dict):
    unnorm_weights = np.array([weights_dict[label] for label in labels])
    weights = unnorm_weights / unnorm_weights.sum()
    classes_roc_auc = roc_auc_score(y_true, y_pred, labels=labels,
                                    multi_class="ovr", average=None)
    return sum(weights * classes_roc_auc)


data_path = "data/train_data_xgbfilled.pqt"
train_df = pd.read_parquet(data_path)
labels, x, y = load_data.extract_month_correlation_features_labels(train_df)

print(x.shape)
print(y.shape)

skf = StratifiedKFold(n_splits=5)

le = LabelEncoder()
le.fit(labels)

results = []
for i, (train_index, val_index) in enumerate(skf.split(x, y)):
    print(f"Fold {i}:")
    x_train = x[train_index]
    x_val = x[val_index]
    y_train = y[train_index]
    y_val = y[val_index]

    # y_train = load_data.one_hot_encoder(y_train, labels)
    # y_val = load_data.one_hot_encoder(y_val, labels)

    y_train = le.transform(y_train)
    y_val = load_data.one_hot_encoder(y_val, labels)

    cluster_weights = pd.read_excel("data/cluster_weights.xlsx").set_index("cluster")
    weights_dict = cluster_weights["unnorm_weight"].to_dict()

    model = xgb.XGBClassifier()
    model.fit(x_train, y_train)

    y_pred_proba = model.predict_proba(x_val)
    result = weighted_roc_auc(y_val, y_pred_proba, labels, weights_dict)
    print(result)
    results.append(result)
print(results)
print(f"cv result: {np.mean(results)}")

