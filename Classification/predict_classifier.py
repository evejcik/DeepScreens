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



df = pd.read_csv("/Users/emmavejcik/Desktop/DeepScreens/Feature_Engineering/Long Long Data.csv")
df = df[~df["film"].isin(["Tron_2059_2148", "Tron_3067_3132"])]

print(df.columns)

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

X = df[ALL_FEATURES]
# y = df[reliability_category_int]

# model = pickle.load(open('reliability_classifier_unweighted.pkl'))

model = joblib.load('reliability_classifier_unweighted.pkl')
y_pred_prob = model.predict_proba(X)[:, 1] 
y_pred = (y_pred_prob > 0.4).astype(int)

df['prob_unreliable'] = y_pred_prob
df['pred_unreliable'] = y_pred

df.to_csv("Predictions Long Data.csv")
print("CSV saved to 'Predictions Long Data.csv' !")

print(f"Shape of dataframe: {df.shape}")
print(df.groupby('joint_name')['joint_id'].unique())