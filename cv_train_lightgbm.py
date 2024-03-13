import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def weighted_roc_auc(y_true, y_pred, labels, weights_dict):
    unnorm_weights = np.array([weights_dict[label] for label in labels])
    weights = unnorm_weights / unnorm_weights.sum()
    classes_roc_auc = roc_auc_score(y_true, y_pred, labels=labels,
                                    multi_class="ovr", average=None)
    return sum(weights * classes_roc_auc)


train_df = pd.read_parquet("data/train_data.pqt")
cat_cols = [
    "channel_code", "city", "city_type",
    "okved", "segment", "start_cluster",
    "index_city_code", "ogrn_month", "ogrn_year",
]
train_df[cat_cols] = train_df[cat_cols].astype("category")

X = train_df.drop(["id", "date", "end_cluster"], axis=1)
y = train_df["end_cluster"].to_numpy()

skf = StratifiedKFold(n_splits=5)

results = []
for i, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    x_train = X.iloc[train_index]
    x_val = X.iloc[val_index]
    y_train = y[train_index]
    y_val = y[val_index]

    model = LGBMClassifier(verbosity=-1, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

    cluster_weights = pd.read_excel("data/cluster_weights.xlsx").set_index("cluster")
    weights_dict = cluster_weights["unnorm_weight"].to_dict()

    y_pred_proba = model.predict_proba(x_val)
    print(y_pred_proba.shape)

    results.append(weighted_roc_auc(y_val, y_pred_proba, model.classes_, weights_dict))
print(results)
print(f"cv result: {np.mean(results)}")
