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
import joblib
import pickle

#train features
#test features

FILMS = [
    "Moonlight_1_1529",
    "Ramona_1_1639",
    "Tron_2059_2148",
    "Tron_3067_3132",
    "Psycho_319_1411",
    "Psycho_319_2006",
]

# Full feature set: anything used by at least one ablation.
ALL_FEATURES = [
    "joint_id",
    "mmpose_confidence",
    "dist_to_boundary",
    "bone_ratio",
    "bone_length",
    "geom_plausible",
    "confidence_std_wk",
    "position_velocity",
    "position_acceleration",
    "position_std_x_wk",
    "position_std_y_wk"
]

TARGET = 'reliability_category_int'

df = pd.read_csv("/Users/emmavejcik/Desktop/DeepScreens/Feature_Engineering/Long Data.csv")

df = df[~df["film"].isin(["Tron_2059_2148", "Tron_3067_3132"])]

X = df[ALL_FEATURES]
y = df['reliability_category_int']

clf = lgb.LGBMClassifier(class_weight=None, random_state=0, n_estimators=200, n_jobs=-1, verbose=-1)
clf.fit(X, y)

joblib.dump(clf, "reliability_classifier_unweighted.pkl")

print(f"Done training model! Model dumped at reliability_classifier_unweighted.pkl.")


