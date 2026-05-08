import pandas as pd
import numpy as np
import joblib

INPUT_CSV     = "/Users/emmavejcik/Desktop/DeepScreens/Feature_Engineering/Long Long Data.csv"  # the H36M-remapped one
MODEL_PATH    = "reliability_classifier_unweighted.pkl"
OUTPUT_CSV    = "Predictions Long Data.csv"

ALL_FEATURES = [
    "joint_id", "mmpose_confidence", "dist_to_boundary",
    "bone_ratio", "bone_length", "geom_plausible",
    "confidence_std_wk", "position_velocity", "position_acceleration",
    "position_std_x_wk", "position_std_y_wk",
]

# Load and filter
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"Loaded {len(df)} rows from {INPUT_CSV}")

# Optional: exclude Tron at inference time too
df = df[~df["film"].isin(["Tron_2059_2148", "Tron_3067_3132"])].reset_index(drop=True)
print(f"After excluding Tron: {len(df)} rows")

# Feature prep — must match training exactly
for col in ALL_FEATURES:
    if df[col].dtype == "object":
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(-1)

X = df[ALL_FEATURES]

# Predict
model = joblib.load(MODEL_PATH)
dt_idx = list(model.classes_).index(2)
df["prob_unreliable"] = model.predict_proba(X)[:, dt_idx]

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {OUTPUT_CSV}")

# Diagnostics
print(f"\nProbability distribution:")
print(df["prob_unreliable"].describe())
print(f"\nFlag rate by joint at threshold 0.5:")
print((df["prob_unreliable"]  > 0.5).groupby(df["joint_name"]).mean().to_string())

joint_film = (
    df.groupby(['film', 'joint_name'])['prob_unreliable']
    .mean()
    .reset_index()
    .rename(columns={'prob_unreliable': 'mean_prob_unreliable'})
)

for film, group in joint_film.groupby('film'):
    ranked = group.sort_values('mean_prob_unreliable', ascending=False)
    print(f"\n{film}")
    print(f"  Most unreliable:")
    for _, row in ranked.head(2).iterrows():
        print(f"    {row['joint_name']:<20} {row['mean_prob_unreliable']:.4f}")
    print(f"  Most reliable:")
    for _, row in ranked.tail(2).iterrows():
        print(f"    {row['joint_name']:<20} {row['mean_prob_unreliable']:.4f}")