import pandas as pd
df = pd.read_csv("Long Data.csv")

# Per-joint correlation
# per_joint = df.groupby('joint_name').apply(
#     lambda g: g[['reliability_category_int', 'mmpose_confidence']].corr().iloc[0, 1]
# ).sort_values()

# print("Per-joint correlation between mmpose_confidence and reliability_category_int:")
# print(per_joint.to_string())
# print(f"\nMedian per-joint correlation: {per_joint.median():.3f}")
# print(f"Mean per-joint correlation:   {per_joint.mean():.3f}")
# print(f"Pooled correlation (all joints): "
#       f"{df[['reliability_category_int','mmpose_confidence']].corr().iloc[0,1]:.3f}")

# # How many rows per joint? Tiny groups produce noisy correlations.
# print("\nRow counts per joint:")
# print(df['joint_name'].value_counts())

# import pandas as pd
# df = pd.read_csv("Long Data.csv", low_memory=False)
# print("dtype:", df['dist_to_boundary'].dtype)
# non_numeric = pd.to_numeric(df['dist_to_boundary'], errors='coerce').isna() & df['dist_to_boundary'].notna()
# print(f"Non-numeric rows: {non_numeric.sum()}")
# print(df.loc[non_numeric, ['film', 'frame_id', 'joint_name', 'dist_to_boundary']].head(20))
# print("Unique non-numeric values:", df.loc[non_numeric, 'dist_to_boundary'].unique()[:20])


# df = pd.read_csv("Long Long Data.csv")

# print(f"Films: {df['film'].unique()}")
# print(df.groupby('film')['mmpose_confidence'].mean())

# print(df.iloc[:, 12])

# df = pd.read_csv("Long Long Data.csv", low_memory=False)
# print(df.groupby('joint_id')['geom_plausible'].apply(lambda s: s.notna().sum()))
# print(df['geom_plausible'].value_counts(dropna=False))

# df_g = df[df['geom_plausible'] != -1]
# print(df_g['joint_name'].unique())

import pandas as pd
df = pd.read_csv("Long Data.csv", low_memory=False)
# df = pd.read_csv("Long Long Data.csv", low_memory = False)
# trust = df[df['reliability_category_int'] == 0]
# trust = trust[trust['bone_ratio'] != -1]
# trust = trust[trust['bone_ratio'] != '-1']  # in case it's a string

# # bone_ratio in Long Data.csv was computed by the OLD geom code, so it only
# # has values for the 5-bone subset. We want bounds keyed by joint_name.
# print(f"Trust rows with real bone_ratio: {len(trust)}")
# print(f"\nPer-joint bounds (5th, 95th percentile of bone_ratio on trust rows):")
# for joint in sorted(trust['joint_name'].unique()):
#     rows = trust[trust['joint_name'] == joint]
#     if len(rows) < 30:
#         print(f"  {joint:18s} n={len(rows):5d}  TOO FEW ROWS")
#         continue
#     bone_ratio = pd.to_numeric(rows['bone_ratio'], errors='coerce').dropna()
#     p05 = bone_ratio.quantile(0.05)
#     p95 = bone_ratio.quantile(0.95)
#     median = bone_ratio.median()
#     print(f"  {joint:18s} n={len(rows):5d}  median={median:.2f}  bounds=({p05:.2f}, {p95:.2f})")

# # # df = pd.read_csv("Long Data.csv", low_memory=False)
# df['bone_ratio'] = pd.to_numeric(df['bone_ratio'], errors='coerce')
# real = df[(df['reliability_category_int'] == 0) & (df['bone_ratio'] != -1)]
# print("geom_flag distribution on trust rows with real bone_ratio:")
# print(real['geom_flag'].value_counts().head(20))

# mask = (df['joint_name'] == 'left_shoulder') & (df['bone_ratio'] != '-1') & (df['bone_ratio'] != -1)
# print(df[mask][['film', 'joint_id', 'joint_name', 'bone_ratio', 'geom_flag']].head(10))
# print()
# print(df[mask]['joint_id'].value_counts())


# # H36M id should match joint_name. Check:
# H36M_NAME_TO_ID = {
#     'right_hip': 1, 'right_knee': 2, 'right_ankle': 3,
#     'left_hip': 4, 'left_knee': 5, 'left_ankle': 6,
#     'spine': 7, 'thorax': 8, 'neck_base': 9, 'head': 10,
#     'left_shoulder': 11, 'left_elbow': 12, 'left_wrist': 13,
#     'right_shoulder': 14, 'right_elbow': 15, 'right_wrist': 16,
#     'root': 0,
# }

# df['expected_h36m_id'] = df['joint_name'].map(H36M_NAME_TO_ID)
# df['stored_id'] = pd.to_numeric(df['joint_id'], errors='coerce')
# mismatched = df[df['stored_id'] != df['expected_h36m_id']]
# print(f"Total rows: {len(df)}")
# print(f"Mismatched: {len(mismatched)}")
# print(f"\nBy film:")
# print(mismatched['film'].value_counts())
# print(f"\nBy joint_name:")
# print(mismatched.groupby(['joint_name', 'stored_id']).size())

print(df.groupby('joint_name')['joint_id'].unique())
print(df.shape)

# df[df['joint_name'] == head, ]

import pandas as pd
import joblib

# Load model and inference data
model = joblib.load("../Classification/reliability_classifier_unweighted.pkl")
df = pd.read_csv("../Classification/Predictions Long Data.csv", low_memory=False)

# Filter to Ramona head
ramona_head = df[(df['film'] == 'Ramona_1_1639') & (df['joint_name'] == 'head')]
print(f"Ramona head inference rows: {len(ramona_head)}")
print(f"Mean prob_unreliable: {ramona_head['prob_unreliable'].mean():.3f}")
print(f"Median prob_unreliable: {ramona_head['prob_unreliable'].median():.3f}")
print(f"% > 0.5: {(ramona_head['prob_unreliable'] > 0.5).mean():.1%}")

# Compare feature distributions: Ramona head (inference) vs all head training data
training = pd.read_csv("Long Data.csv", low_memory=False)
training_head = training[(training['joint_name'] == 'head') & 
                          (training['reliability_category_int'].notna())]

FEATURES = ["mmpose_confidence", "bone_ratio", "geom_plausible",
            "position_velocity", "position_std_x_wk"]

print("\nFeature comparison (training head trust vs Ramona head inference):")
print(f"{'Feature':<25s} {'Train (trust)':>15s} {'Ramona (inf)':>15s}")
for col in FEATURES:
    train_trust = training_head[training_head['reliability_category_int'] == 0][col].mean()
    ramona_inf = pd.to_numeric(ramona_head[col], errors='coerce').mean()
    print(f"{col:<25s} {train_trust:>15.3f} {ramona_inf:>15.3f}")

