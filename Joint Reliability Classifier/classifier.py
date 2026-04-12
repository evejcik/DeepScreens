import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
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
    'position_velocity',
    'position_acceleration',
    'position_std_x_wk',
    'position_std_y_wk',
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
PSYCHO_JOINTS = ['left_elbow', 'right_hip']

# Class weights: inverse frequency, emphasising dont_trust precision
# trust=0 (69.6%), partial_trust=1 (6.5%), dont_trust=2 (23.4%)
# CLASS_WEIGHTS = {0: 1.0, 1: 4.0, 2: 2.0}
CLASS_WEIGHTS = {'Moonlight_1_1529': 1.0, 'Ramona_1_1639': 4.0, 'Tron_2059_2148' : 2.0} #for legibility

# ── 1. Prepare data ───────────────────────────────────────────────────────────

def cross_validation(df, k = 3):
    #k is number of unique films we're passing in here
    data = pd.read_csv(df)
    train_data = data[data['film_id'].isin(TRAIN_FILM_IDS)]
    test_data = data[(data['film_id'].isin(TEST_FILM_IDS)) & (data['joint_name'].isin(PSYCHO_JOINTS))]

    X, y = train_data[FEATURES], train_data[TARGET]
    groups = train_data['film'] #putting film name not film id now just for legibility -> now groups is a series!
    group_kfold = GroupKFold(n_splits = k)
    scaler = StandardScaler()
    #GroupKFold.split() needs to know which group every single row belongs to, so it can ensure all rows from the same film stay together
    print(groups)
    
    for i, (train_index, val_index) in enumerate(group_kfold.split(X, y, groups)):
        train_films = groups.iloc[train_index].unique()
        val_films = groups.iloc[val_index].unique()
        print(f"Fold {i}: train={train_films}, val={val_films}")

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        ###log regression TRAIN & VALIDATE
        clf = LogisticRegression(random_state = 0, class_weight = 'balanced').fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        print(classification_report(y_val, y_pred))

        ###log regression TEST
        


    
    # print(f"Score: {clf.score(X, y)}\n")




# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# def prepare(df: pd.DataFrame):
#     """
#     Merge ambiguous into partial_trust, encode geom_plausible,
#     encode film_id, return clean df.
#     """
#     df = df.copy()

#     # Merge ambiguous (3) into partial_trust (1)
#     df[TARGET] = df[TARGET].replace(3, 1)

#     # Encode geom_plausible: True→1, False→0, -1→-1 (unknown)
#     df['geom_plausible'] = df['geom_plausible'].map(
#         {True: 1, 1: 1, False: 0, 0: 0, -1: -1}
#     ).fillna(-1).astype(int)

#     # Encode film_id if not already done
#     if 'film_id' not in df.columns:
#         film_map = {f: i for i, f in enumerate(df['film'].unique())}
#         df['film_id'] = df['film'].map(film_map)

#     # Fill any remaining NaNs in features with -1 sentinel
#     df[FEATURES] = df[FEATURES].fillna(-1)

#     return df


# def temporal_block_split(df: pd.DataFrame, test_frac: float = 0.2):
#     """
#     For within-film validation: split by taking the last test_frac of
#     frame_ids per (film, instance_id, joint_name) group as test.
#     Never split randomly — leaks temporal context.
#     """
#     train_idx, val_idx = [], []
#     for _, grp in df.groupby(['film', 'instance_id', 'joint_name']):
#         grp_sorted = grp.sort_values('frame_id')
#         n = len(grp_sorted)
#         split = int(n * (1 - test_frac))
#         train_idx.extend(grp_sorted.index[:split].tolist())
#         val_idx.extend(grp_sorted.index[split:].tolist())
#     return train_idx, val_idx


# # ── 2. Evaluation ─────────────────────────────────────────────────────────────

# def evaluate(model, X, y, label: str, tier_joints: dict = None,
#              joint_ids=None):
#     """
#     Full evaluation: classification report + confusion matrix.
#     Optionally break down dont_trust precision by joint tier.
#     """
#     preds = model.predict(X)
#     print(f"\n{'='*60}")
#     print(f"EVALUATION: {label}")
#     print(f"{'='*60}")
#     print(classification_report(
#         y, preds,
#         target_names=['trust', 'partial_trust', 'dont_trust'],
#         digits=3
#     ))

#     cm = confusion_matrix(y, preds)
#     disp = ConfusionMatrixDisplay(
#         cm, display_labels=['trust', 'partial_trust', 'dont_trust']
#     )
#     fig, ax = plt.subplots(figsize=(6, 5))
#     disp.plot(ax=ax, cmap='Blues')
#     ax.set_title(f'Confusion Matrix — {label}')
#     plt.tight_layout()
#     plt.savefig(f'confusion_matrix_{label.replace(" ", "_")}.png', dpi=150)
#     plt.close()

#     # Per-tier dont_trust precision if joint_ids provided
#     if tier_joints and joint_ids is not None:
#         print("\nDont-trust precision by joint tier:")
#         for tier, joints in tier_joints.items():
#             mask = np.isin(joint_ids, joints)
#             if mask.sum() == 0:
#                 continue
#             t_true = y[mask]
#             t_pred = preds[mask]
#             tp = ((t_pred == 2) & (t_true == 2)).sum()
#             fp = ((t_pred == 2) & (t_true != 2)).sum()
#             prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#             rec  = tp / (t_true == 2).sum() if (t_true == 2).sum() > 0 else 0.0
#             print(f"  {tier}: precision={prec:.3f}  recall={rec:.3f}  "
#                   f"n_samples={mask.sum()}")


# # ── 3. Feature importance ─────────────────────────────────────────────────────

# def plot_importance(model, feature_names: list, label: str):
#     if hasattr(model, 'feature_importances_'):
#         imp = model.feature_importances_
#     elif hasattr(model, 'coef_'):
#         imp = np.abs(model.coef_).mean(axis=0)
#     else:
#         return

#     idx = np.argsort(imp)[::-1]
#     plt.figure(figsize=(8, 5))
#     plt.bar(range(len(feature_names)), imp[idx])
#     plt.xticks(range(len(feature_names)),
#                [feature_names[i] for i in idx], rotation=45, ha='right')
#     plt.title(f'Feature Importance — {label}')
#     plt.tight_layout()
#     plt.savefig(f'feature_importance_{label.replace(" ", "_")}.png', dpi=150)
#     plt.close()
#     print(f"\nTop features ({label}):")
#     for i in idx[:5]:
#         print(f"  {feature_names[i]}: {imp[i]:.4f}")


# # ── 4. Main ───────────────────────────────────────────────────────────────────

# def main(df: pd.DataFrame):

#     df = prepare(df)

#     # ── 4a. Train / test split ────────────────────────────────────────────────
#     train_df = df[df['film'].isin(TRAIN_FILMS)].copy()
#     test_df  = df[df['film'] == TEST_FILM].copy()

#     # Psycho: only evaluate on joints shared with training films
#     test_df_shared = test_df[test_df['joint_name'].isin(PSYCHO_JOINTS)].copy()

#     # Within-training temporal block split for validation
#     train_idx, val_idx = temporal_block_split(train_df)
#     val_df   = train_df.loc[val_idx].copy()
#     train_df = train_df.loc[train_idx].copy()

#     print(f"\nData split:")
#     print(f"  Train:      {len(train_df):>6} rows")
#     print(f"  Val:        {len(val_df):>6} rows  (temporal block, within training films)")
#     print(f"  Test:       {len(test_df):>6} rows  (Psycho, all joints)")
#     print(f"  Test shared:{len(test_df_shared):>6} rows  (Psycho, left_elbow + right_hip only)")

#     X_train = train_df[FEATURES].values
#     y_train = train_df[TARGET].values.astype(int)
#     w_train = train_df[TARGET].map(CLASS_WEIGHTS).values

#     X_val   = val_df[FEATURES].values
#     y_val   = val_df[TARGET].values.astype(int)

#     X_test  = test_df_shared[FEATURES].values
#     y_test  = test_df_shared[TARGET].values.astype(int)
#     jids_test = test_df_shared['joint_id'].values

#     # ── 4b. Tier definitions for per-tier evaluation ──────────────────────────
#     # H36M joint IDs by tier
#     TIER_JOINTS = {
#         'Tier1_hips_shoulders': [1, 4, 11, 14],
#         'Tier2_limbs':          [2, 3, 5, 6, 12, 13, 15, 16],
#     }

#     # ── 4c. Baseline: Logistic Regression ────────────────────────────────────
#     print("\n" + "="*60)
#     print("BASELINE: Logistic Regression")
#     print("="*60)

#     lr_pipe = Pipeline([
#         ('scaler', StandardScaler()),
#         ('clf', LogisticRegression(
#             class_weight=CLASS_WEIGHTS,
#             max_iter=1000,
#             random_state=42,
#             multi_class='multinomial'
#         ))
#     ])
#     lr_pipe.fit(X_train, y_train, clf__sample_weight=w_train)

#     evaluate(lr_pipe, X_val,  y_val,  'LR — Validation')
#     evaluate(lr_pipe, X_test, y_test, 'LR — Psycho Test',
#              TIER_JOINTS, jids_test)
#     plot_importance(lr_pipe.named_steps['clf'], FEATURES, 'LogisticRegression')

#     # ── 4d. Main model: LightGBM ──────────────────────────────────────────────
#     print("\n" + "="*60)
#     print("MAIN MODEL: LightGBM")
#     print("="*60)

#     lgb_model = lgb.LGBMClassifier(
#         n_estimators=500,
#         learning_rate=0.05,
#         num_leaves=31,
#         class_weight=CLASS_WEIGHTS,
#         random_state=42,
#         n_jobs=-1,
#         verbose=-1,
#     )
#     lgb_model.fit(
#         X_train, y_train,
#         sample_weight=w_train,
#         eval_set=[(X_val, y_val)],
#         callbacks=[lgb.early_stopping(50, verbose=False),
#                    lgb.log_evaluation(100)]
#     )

#     evaluate(lgb_model, X_val,  y_val,  'LGB — Validation')
#     evaluate(lgb_model, X_test, y_test, 'LGB — Psycho Test',
#              TIER_JOINTS, jids_test)
#     plot_importance(lgb_model, FEATURES, 'LightGBM')

#     # ── 4e. Probability outputs for downstream cleaning ───────────────────────
#     # The cleaning pipeline needs P(dont_trust) per joint, not hard labels
#     # Save probability outputs on the full dataset for inspection
#     df_out = df.copy()
#     X_all = df_out[FEATURES].fillna(-1).values
#     probs = lgb_model.predict_proba(X_all)  # (n, 3): [P(trust), P(pt), P(dt)]
#     df_out['p_trust']        = probs[:, 0]
#     df_out['p_partial_trust'] = probs[:, 1]
#     df_out['p_dont_trust']   = probs[:, 2]
#     df_out['pred_label']     = lgb_model.predict(X_all)

#     # Flag: use hard threshold on P(dont_trust) for cleaning pipeline
#     DONT_TRUST_THRESHOLD = 0.5  # tune this based on precision/recall tradeoff
#     df_out['filter_flag'] = (df_out['p_dont_trust'] > DONT_TRUST_THRESHOLD).astype(int)

#     print(f"\nFilter flag distribution:")
#     print(df_out['filter_flag'].value_counts())
#     print(f"\nOf flagged joints, fraction truly dont_trust: "
#           f"{(df_out[df_out['filter_flag']==1][TARGET]==2).mean():.3f}")

#     df_out.to_csv('reliability_predictions.csv', index=False)
#     print("\nSaved: reliability_predictions.csv")

#     return lgb_model, df_out


# # ── 5. Entry point ────────────────────────────────────────────────────────────
# # Call main(df) after running your feature engineering script
# # e.g.:
# #   from feature_engineering import main as build_features
# #   df = build_features(csv_path, k=5)
# #   model, results = main(df)

def main(df):
    cross_validation(df)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")

    args = ap.parse_args()
    main(
        args.csv
    )