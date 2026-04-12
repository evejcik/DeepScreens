import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import argparse

# ── 0. Constants ──────────────────────────────────────────────────────────────

FEATURES = [
    'joint_id',
    'mmpose_confidence',
    'dist_to_boundary',
    'bone_ratio',
    'bone_length',
    'geom_plausible',
    'confidence_std_wk',
    # 'position_velocity',
    # 'position_acceleration',
    # 'position_std_x_wk',
    # 'position_std_y_wk',
    'frames_since_trust',
]

TARGET = 'reliability_category_int'

# Films in training set — Psycho is held-out test
TRAIN_FILMS = ['Moonlight_1_1529', 'Ramona_1_1639', 'Tron_2059_2148']
TRAIN_FILM_IDS = [0, 2, 3]
TEST_FILM   = 'Psycho_319_1411'
TEST_FILM_IDS = [1]

# Joints that appear in both Psycho and at least one training film
# Only evaluate generalization on these joints
PSYCHO_JOINTS = ['left_elbow', 'right_hip', 'left_hip', 'left_shoulder', 'right_shoulder']

# Class weights: inverse frequency, emphasising dont_trust precision
# trust=0 (69.6%), partial_trust=1 (6.5%), dont_trust=2 (23.4%)
CLASS_WEIGHTS = {0: 1.0, 1: 4.0, 2: 2.0}
# CLASS_WEIGHTS = {'Moonlight_1_1529': 1.0, 'Ramona_1_1639': 4.0, 'Tron_2059_2148' : 2.0} #for legibility

# ── 1. Prepare data ───────────────────────────────────────────────────────────

def lightGBM_clf(df, k = 3):
    #k is number of unique films we're passing in here
    data = pd.read_csv(df)
    print(data.isnull().sum()[data.isnull().sum() > 0])
    train_data = data[data['film_id'].isin(TRAIN_FILM_IDS)]
    test_data = data[(data['film_id'].isin(TEST_FILM_IDS)) & (data['joint_name'].isin(PSYCHO_JOINTS))]

    X, y = train_data[FEATURES], train_data[TARGET]
    X_test, y_test = test_data[FEATURES], test_data[TARGET]
    groups = train_data['film'] #putting film name not film id now just for legibility -> now groups is a series!
    group_kfold = GroupKFold(n_splits = k)
    scaler = StandardScaler()
    #GroupKFold.split() needs to know which group every single row belongs to, so it can ensure all rows from the same film stay together
    # print(groups)
    
    for i, (train_index, val_index) in enumerate(group_kfold.split(X, y, groups)):
        train_films = groups.iloc[train_index].unique()
        val_films = groups.iloc[val_index].unique()
        print(f"Fold {i}: train={train_films}, val={val_films}")

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        sample_weights = y_train.map(CLASS_WEIGHTS)


        # X_train = scaler.fit_transform(X_train)
        # X_val = scaler.transform(X_val)

        lgb_train = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        lgb_valid = lgb.Dataset(X_test, label = y_test, reference = lgb_train)
        ###lLGBM

        params={
            "objective":"multiclass",
            "metric":["multi_logloss","multi_error"],
            "boosting_type":"gbdt",
            "num_class" : 3,
            "learning_rate":0.05,
            "num_leaves":31,
            "max_depth":-1,
            "feature_fraction":0.8,
            "bagging_fraction":0.8,
            "bagging_freq":5,
            "lambda_l1":0.1,
            "lambda_l2":0.2,
            "min_data_in_leaf":20,
            "verbose":-1
            }

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train,lgb_valid],
            valid_names=["train","valid"],
            callbacks=[lgb.early_stopping(50)]
        )
        
        y_pred_prob=model.predict(X_test,num_iteration=model.best_iteration)
        y_pred = y_pred_prob.argmax(axis=1)

        accuracy=accuracy_score(y_test,y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        # auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

        print("Accuracy:",accuracy)
        print("Precision:",precision)
        print("Recall:",recall)
        print("F1 Score:",f1)
        # print("AUC:",auc)

        print(f"--- Psycho test (fold {i}) ---")
        print(classification_report(y_test, y_pred))


def main(df):
    lightGBM_clf(df)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")

    args = ap.parse_args()
    main(
        args.csv
    )