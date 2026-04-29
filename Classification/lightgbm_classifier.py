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


# ── 0. Constants ──────────────────────────────────────────────────────────────

FEATURES = [
    'joint_id',
    'mmpose_confidence',
    'dist_to_boundary',
    'bone_ratio',
    'bone_length',
    'geom_plausible',
    'confidence_std_wk'
    # 'position_velocity',
    # 'position_acceleration',
    # 'position_std_x_wk',
    # 'position_std_y_wk',
    # 'frames_since_trust', #we discovered that these features were not actually that useful I believe
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
# CLASS_WEIGHTS = {0: 1.0, 1: 4.0, 2: 2.0}
CLASS_WEIGHTS = {0: 1.0, 1: 2.0}
# CLASS_WEIGHTS = {'Moonlight_1_1529': 1.0, 'Ramona_1_1639': 4.0, 'Tron_2059_2148' : 2.0} #for legibility

RELEVANT_JOINT_IDS = list(range(17))  # COCO body joints only, no face mesh / fingers

PARAMS = {
    "objective":        "binary",
    "metric":           ["auc", "binary_logloss"],
    "boosting_type":    "gbdt",
    "learning_rate":    0.05,
    "num_leaves":       31,
    "max_depth":        -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "lambda_l1":        0.1,
    "lambda_l2":        0.2,
    "min_data_in_leaf": 20,
    "verbose":          -1,
}

# ── 1. Prepare data ───────────────────────────────────────────────────────────

def load_and_prepare(csv_path):
    data = pd.read_csv(csv_path)

    # Binarise: trust=0, partial+dont_trust=1; leave -1 (unannotated) untouched

    annotated_mask = data[TARGET].notna()
    data.loc[annotated_mask, TARGET] = data.loc[annotated_mask, TARGET].replace({0: 0, 1: 1, 2: 1})

    print(f"Loaded {len(data)} rows.")
    print(f"  Annotated:   {annotated_mask.sum()}")
    print(f"  Unannotated: {(~annotated_mask).sum()}")
    print(f"  Null counts:\n{data.isnull().sum()[data.isnull().sum() > 0]}\n")

    return data

def cross_validate(X, y, groups, X_test, y_test, k=3):
    group_kfold  = GroupKFold(n_splits=k)
    rows         = []
    best_iters   = []

    for i, (train_index, val_index) in enumerate(group_kfold.split(X, y, groups)):
        train_films = groups.iloc[train_index].unique()
        val_films   = groups.iloc[val_index].unique()
        print(f"\nFold {i}: train={train_films}, val={val_films}")

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        sample_weights = y_train.map(CLASS_WEIGHTS)

        lgb_train = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        lgb_val_set = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        model = lgb.train(
            PARAMS,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_val_set],
            valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(50)],
        )

        best_iters.append(model.best_iteration)

        # Evaluate on held-out Psycho test set
        y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred      = (y_pred_prob >= 0.5).astype(int)

        print(f"Fold {i} — Psycho test set:")
        print(classification_report(y_test, y_pred))
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  Feature importances: {sorted(zip(model.feature_name(), model.feature_importance()), key=lambda x: x[1], reverse=True)}")

        # Store val predictions (out-of-fold, unseen by this fold's model)
        y_pred_prob_val = model.predict(X_val, num_iteration=model.best_iteration)
        for original_ind, prob in zip(X_val.index, y_pred_prob_val):
            rows.append({'original_index': original_ind, 'prob_unreliable': prob})

    mean_best_iter = int(np.mean(best_iters))
    print(f"\nMean best iteration across folds: {mean_best_iter}")
    return rows, mean_best_iter

def train_final_model(X, y, num_boost_round):
    sample_weights = y.map(CLASS_WEIGHTS)
    lgb_train_full = lgb.Dataset(X, label=y, weight=sample_weights)

    model = lgb.train(
        PARAMS,
        lgb_train_full,
        num_boost_round=num_boost_round,
    )
    return model

def run_inference(model, data, rows):
    """
    Predict prob_unreliable for:
      - annotated test rows (Psycho)
      - unannotated rows for relevant joints (0-16) only
    Appends results to rows and returns updated rows.
    """
    # Psycho test set (annotated)
    test_data = data[
        (data['film_id'].isin(TEST_FILM_IDS)) &
        (data[TARGET].notna())
    ]
    X_test  = test_data[FEATURES]
    y_test  = test_data[TARGET]

    psycho_probs = model.predict(X_test, num_iteration=model.best_iteration)
    psycho_preds = (psycho_probs >= 0.5).astype(int)

    print("\n--- Final model: Psycho test (all training films) ---")
    print(classification_report(y_test, psycho_preds))
    print(f"Accuracy:  {accuracy_score(y_test, psycho_preds):.3f}")
    print(f"Precision: {precision_score(y_test, psycho_preds, average='macro'):.3f}")
    print(f"Recall:    {recall_score(y_test, psycho_preds, average='macro'):.3f}")
    print(f"F1:        {f1_score(y_test, psycho_preds, average='macro'):.3f}")

    for original_ind, prob in zip(X_test.index, psycho_probs):
        rows.append({'original_index': original_ind, 'prob_unreliable': prob})

    # Unannotated rows — restrict to relevant joints only
    unannotated_mask = (
        data[TARGET].isna() &
        (data['joint_id'].isin(RELEVANT_JOINT_IDS))
    )
    X_unann = data.loc[unannotated_mask, FEATURES].copy()
    X_unann = X_unann.fillna(-1)  # LightGBM handles -1 as missing natively
    X_unann_valid = X_unann  # all rows are now valid

    if len(X_unann_valid) == 0:
        print("WARNING: no valid unannotated rows to predict on.")
        return rows

    print(f"\nUnannotated rows eligible for inference: {unannotated_mask.sum()}")
    print(f"  All rows will be predicted (NaNs filled with -1).")

    unann_probs = model.predict(X_unann_valid, num_iteration=model.best_iteration)
    for original_ind, prob in zip(X_unann_valid.index, unann_probs):
        rows.append({'original_index': original_ind, 'prob_unreliable': prob})

    return rows

def lightGBM_clf(csv_path, k=3):
    data = load_and_prepare(csv_path)

    annotated_mask = data[TARGET].notna()
    train_data = data[data['film_id'].isin(TRAIN_FILM_IDS) & annotated_mask]
    test_data  = data[
        data['film_id'].isin(TEST_FILM_IDS) &
        # data['joint_name'].isin(PSYCHO_JOINTS) &
        annotated_mask
    ]

    print(f"Train rows: {len(train_data)}")
    print(f"Test rows:  {len(test_data)}")
    print(f"Test target distribution:\n{test_data[TARGET].value_counts()}\n")

    X, y   = train_data[FEATURES], train_data[TARGET]
    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]
    groups = train_data['film']

    # Cross-validation
    rows, mean_best_iter = cross_validate(X, y, groups, X_test, y_test, k=k)

    # Final model on all training data
    model = train_final_model(X, y, num_boost_round=mean_best_iter)

    # Inference on Psycho + unannotated
    rows = run_inference(model, data, rows)

    # Merge probabilities back into data
    probs_df = pd.DataFrame(rows).set_index('original_index')
    probs_df = probs_df[~probs_df.index.duplicated(keep='last')]  # Psycho appears in both cv and final
    data['prob_unreliable'] = probs_df['prob_unreliable']

    psycho_unann_body = data[
        data['film_id'].isin(TEST_FILM_IDS) & 
        data[TARGET].isna() & 
        data['joint_id'].isin(RELEVANT_JOINT_IDS)
    ]
    print(f"Psycho unannotated body joints: {len(psycho_unann_body)}")
    print(f"Of those, prob_unreliable null: {psycho_unann_body['prob_unreliable'].isna().sum()}")

    print(f"\nprob_unreliable null count: {data['prob_unreliable'].isna().sum()}")
    print(f"Psycho null count:          {data[data['film_id'].isin(TEST_FILM_IDS)]['prob_unreliable'].isna().sum()}")

    data.to_csv('../Feature_Engineering/Long_Long_Data_with_probs.csv', index=False)
    print("Saved to Feature_Engineering/Long_Long_Data_with_probs.csv")

    joblib.dump(model, 'lgbm_reliability_model.pkl')
    print("Model saved to lgbm_reliability_model.pkl")

def ablation_studies(csv_path):
    data = load_and_prepare(csv_path)

    annotated_mask = data[TARGET].notna()
    train_data = data[data['film_id'].isin(TRAIN_FILM_IDS) & annotated_mask]
    test_data  = data[
        data['film_id'].isin(TEST_FILM_IDS) &
        data['joint_name'].isin(PSYCHO_JOINTS) &
        annotated_mask
    ]

    y_train = train_data[TARGET]
    y_test  = test_data[TARGET]
    groups  = train_data['film']

    feature_sets = {
        'confidence_only':  ['mmpose_confidence'],
        'geometric_only':   ['dist_to_boundary', 'bone_ratio', 'bone_length', 'geom_plausible', 'joint_id'],
        'all_features':     FEATURES,
    }

    results = {}

    for name, features in feature_sets.items():
        print(f"\n{'='*60}")
        print(f"ABLATION: {name}")
        print(f"Features: {features}")
        print(f"{'='*60}")

        X_train = train_data[features]
        X_test  = test_data[features]

        # get mean best iter via CV
        group_kfold = GroupKFold(n_splits=3)
        best_iters  = []

        for i, (train_idx, val_idx) in enumerate(group_kfold.split(X_train, y_train, groups)):
            Xt, Xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            yt, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]

            sample_weights = yt.map(CLASS_WEIGHTS)
            lgb_train   = lgb.Dataset(Xt, label=yt, weight=sample_weights)
            lgb_val_set = lgb.Dataset(Xv, label=yv, reference=lgb_train)

            model = lgb.train(
                PARAMS,
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_train, lgb_val_set],
                valid_names=["train", "val"],
                callbacks=[lgb.early_stopping(50)],
            )
            best_iters.append(model.best_iteration)

        best_iter = int(np.median(best_iters))
        print(f"Median best iteration: {best_iter}")

        # final model
        sample_weights = y_train.map(CLASS_WEIGHTS)
        lgb_full = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        final_model = lgb.train(PARAMS, lgb_full, num_boost_round=best_iter)

        # evaluate
        y_pred_prob = final_model.predict(X_test)
        y_pred      = (y_pred_prob >= 0.5).astype(int)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec  = recall_score(y_test, y_pred, average='macro')
        f1   = f1_score(y_test, y_pred, average='macro')

        print(classification_report(y_test, y_pred))
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"F1:        {f1:.3f}")

        results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

        # cases where mmpose is confident but geometric flags as implausible
        if name == 'geometric_only':
            geometric_preds = y_pred
            geometric_probs = y_pred_prob
        if name == 'confidence_only':
            confidence_preds = y_pred

    # summary table
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'-'*65}")
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['accuracy']:>10.3f} {metrics['precision']:>10.3f} "
              f"{metrics['recall']:>10.3f} {metrics['f1']:>10.3f}")

    # cases where geometric catches what confidence misses
    print(f"\n{'='*60}")
    print("DISAGREEMENT ANALYSIS: geometric vs confidence")
    print(f"{'='*60}")
    test_analysis = test_data[['joint_name', 'mmpose_confidence', 
                                'bone_ratio', 'geom_plausible', 
                                'dist_to_boundary', TARGET]].copy()
    test_analysis['confidence_pred'] = confidence_preds
    test_analysis['geometric_pred']  = geometric_preds
    test_analysis['geometric_prob']  = geometric_probs

    # geometric catches it, confidence misses it
    geo_catches = test_analysis[
        (test_analysis['geometric_pred'] == 1) &
        (test_analysis['confidence_pred'] == 0) &
        (test_analysis[TARGET] == 1)
    ]
    # confidence catches it, geometric misses it
    conf_catches = test_analysis[
        (test_analysis['confidence_pred'] == 1) &
        (test_analysis['geometric_pred'] == 0) &
        (test_analysis[TARGET] == 1)
    ]
    # both miss it
    both_miss = test_analysis[
        (test_analysis['confidence_pred'] == 0) &
        (test_analysis['geometric_pred'] == 0) &
        (test_analysis[TARGET] == 1)
    ]

    print(f"Geometric catches, confidence misses: {len(geo_catches)}")
    print(f"Confidence catches, geometric misses: {len(conf_catches)}")
    print(f"Both miss:                            {len(both_miss)}")
    print(f"\nTop joints where geometric uniquely catches failures:")
    print(geo_catches['joint_name'].value_counts().head(10))
    print(f"\nTop joints where confidence uniquely catches failures:")
    print(conf_catches['joint_name'].value_counts().head(10))

    return results

def main(csv_path):
    data = pd.read_csv(csv_path)
    annotated = data[data['reliability_category_int'].notna()]
    print(annotated.groupby('film')['reliability_category_int'].value_counts(normalize=True))
    
    lightGBM_clf(csv_path)
    ablation_studies(csv_path)
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default='../Feature_Engineering/Concatenated_Data.csv')
    ap.add_argument("--k",   type=int, default=3)
    args = ap.parse_args()
    main(args.csv)