import purple.preprocessing.load_data as load_data
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split




data_path = "../../data/train_data.pqt"
train_df = pd.read_parquet(data_path)
x, y = load_data.extract_subj_features_labels(train_df)

x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                  test_size=0.2,
                                                  random_state=42)

