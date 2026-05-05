"""
classifier.py
-------------
Leave-one-film-out cross-validation for joint reliability estimation.

Evaluation protocol
-------------------
6-fold LOFO-CV. Each fold holds out one segment as test, trains on the
other five. No held-out final test set; per project decision (Option A).
Within each fold's training data, the loop runs all four ablations
described below, on identical splits, so their numbers are directly
comparable.

Ablations (same fold structure for all four)
--------------------------------------------
1. baseline:       LogisticRegression, mmpose_confidence ONLY
2. full:           LightGBM, all available features
3. full_no_jid:    LightGBM, all features except joint_id
4. full_no_conf:   LightGBM, all features except mmpose_confidence

Each ablation reports two numbers per fold:
  filtered:   only test rows whose joint_name appears in training
              (fair generalization measurement)
  unfiltered: all test rows including zero-shot joints
              (honest but mixes cross-film and zero-shot generalization)

Headline metric: dont_trust precision (class label = 2). This is the
metric that matters for animation quality - we care about not flagging
good joints as bad. Recall and trust precision are reported as context.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_FILMS = [
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
    "position_std_y_wk",
    # "frames_since_trust",
]

# Per-ablation feature lists; each must be a subset of ALL_FEATURES.
ABLATIONS = {
    "baseline":      ["mmpose_confidence"],
    "full":          ALL_FEATURES,
    "full_no_jid":   [f for f in ALL_FEATURES if f != "joint_id"],
    "full_no_conf":  [f for f in ALL_FEATURES if f != "mmpose_confidence"],
}

TARGET = "reliability_category_int"

DONT_TRUST_LABEL = 2
TRUST_LABEL      = 0


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_and_validate(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    missing = [c for c in ALL_FEATURES + [TARGET, "film", "joint_name"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    n_total = len(df)
    df = df.dropna(subset=[TARGET])
    n_dropped = n_total - len(df)
    if n_dropped:
        print(f"[data] Dropped {n_dropped} rows with NaN target out of {n_total}.")

    # annotator_confidence: not used as a feature, used only as sample weight if present.
    # All other features: fill remaining NaN with sentinel -1 (matches feature engineering).
    for col in ALL_FEATURES:
        n_na = df[col].isna().sum()
        if n_na:
            print(f"[data] Filling {n_na} NaN in {col} with -1.")
            df[col] = df[col].fillna(-1)

    # Coerce target to int and verify only {0, 2} remain (after partial/ambiguous merge).
    df[TARGET] = df[TARGET].astype(int)
    label_set = set(df[TARGET].unique())
    if not label_set.issubset({TRUST_LABEL, DONT_TRUST_LABEL}):
        raise ValueError(f"Unexpected target labels {label_set}; expected subset of "
                         f"{{{TRUST_LABEL}, {DONT_TRUST_LABEL}}}.")

    # Verify all configured films actually appear.
    seen_films = set(df["film"].unique())
    missing_films = [f for f in ALL_FILMS if f not in seen_films]
    if missing_films:
        print(f"[data] WARNING: configured films missing from CSV: {missing_films}")
    extra_films = [f for f in seen_films if f not in ALL_FILMS]
    if extra_films:
        print(f"[data] WARNING: CSV contains films not in ALL_FILMS: {extra_films}")

    return df


# ---------------------------------------------------------------------------
# Per-fold training and evaluation
# ---------------------------------------------------------------------------

def fit_predict_baseline(X_train, y_train, X_test, class_weight):
    """Logistic regression on a single feature with balanced class weights."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    clf = LogisticRegression(class_weight= class_weight, max_iter=1000, random_state=0)
    clf.fit(X_train_s, y_train)
    return clf.predict(X_test_s)


def fit_predict_lightgbm(X_train, y_train, X_test, class_weight):
    """LightGBM with balanced class weights."""
    clf = lgb.LGBMClassifier(
        class_weight= class_weight,
        random_state=0,
        n_estimators=200,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def evaluate(y_true, y_pred):
    """
    Return per-class precision, recall, support, plus confusion matrix.
    Output is a dict indexed by class label.
    """
    labels = [TRUST_LABEL, DONT_TRUST_LABEL]
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "trust_precision":      float(p[0]),
        "trust_recall":         float(r[0]),
        "trust_support":        int(sup[0]),
        "dont_trust_precision": float(p[1]),
        "dont_trust_recall":    float(r[1]),
        "dont_trust_support":   int(sup[1]),
        "confusion_matrix":     cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_lofo_cv(df):
    """
    For each ablation, run LOFO-CV across ALL_FILMS. Within each fold,
    evaluate twice: once on filtered test (joints present in training)
    and once on unfiltered test (all rows). Aggregate results.

    Returns: dict[ablation_name][eval_mode] -> list of per-fold metric dicts.
    """
    results = {ablation: {"filtered": [], "unfiltered": []}
               for ablation in ABLATIONS}
    for class_weight_strategy in [None, "balanced"]:
        print(f"\n{'='*72}")
        print(f"CLASS WEIGHT STRATEGY: {class_weight_strategy}")
        print(f"{'='*72}")
        for test_film in ALL_FILMS:
            if test_film not in df["film"].unique():
                print(f"[fold] SKIP: {test_film} not present in data.")
                continue

            train_df = df[df["film"] != test_film]
            test_df  = df[df["film"] == test_film]
            train_joints = set(train_df["joint_name"].unique())
            test_filtered_df = test_df[test_df["joint_name"].isin(train_joints)]

            zero_shot_joints = sorted(set(test_df["joint_name"].unique()) - train_joints)

            print(f"\n{'='*72}")
            print(f"[fold] Test film: {test_film}")
            print(f"       Train rows: {len(train_df):>6d} ({len(train_df['film'].unique())} films)")
            print(f"       Test rows:  {len(test_df):>6d}  (unfiltered)")
            print(f"                   {len(test_filtered_df):>6d}  (filtered to seen joints)")
            if zero_shot_joints:
                print(f"       Zero-shot joints in test: {zero_shot_joints}")
            print(f"       Test class balance (unfiltered): "
                f"{test_df[TARGET].value_counts().to_dict()}")

            if len(test_filtered_df) == 0:
                print(f"[fold] SKIP: no test rows survive joint filtering.")
                continue

            for ablation, features in ABLATIONS.items():
                X_train = train_df[features]
                y_train = train_df[TARGET]
                X_test_unfilt   = test_df[features]
                y_test_unfilt   = test_df[TARGET]
                X_test_filt     = test_filtered_df[features]
                y_test_filt     = test_filtered_df[TARGET]

                if ablation == "baseline":
                    y_pred_unfilt = fit_predict_baseline(X_train, y_train, X_test_unfilt,class_weight_strategy)
                    y_pred_filt   = fit_predict_baseline(X_train, y_train, X_test_filt,class_weight_strategy)
                else:
                    y_pred_unfilt = fit_predict_lightgbm( X_train, y_train, X_test_unfilt, class_weight_strategy)
                    y_pred_filt   = fit_predict_lightgbm(X_train, y_train, X_test_filt, class_weight_strategy)

                m_unfilt = evaluate(y_test_unfilt, y_pred_unfilt)
                m_filt   = evaluate(y_test_filt,   y_pred_filt)
                m_unfilt["test_film"] = test_film
                m_filt["test_film"]   = test_film
                results[ablation]["unfiltered"].append(m_unfilt)
                results[ablation]["filtered"].append(m_filt)

                print(f"  {ablation:14s}  "
                    f"unfilt: dt_prec={m_unfilt['dont_trust_precision']:.3f} "
                    f"dt_rec={m_unfilt['dont_trust_recall']:.3f}   "
                    f"filt:   dt_prec={m_filt['dont_trust_precision']:.3f} "
                    f"dt_rec={m_filt['dont_trust_recall']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize_results(results):
    """Build a wide DataFrame: rows = (ablation, eval_mode), cols = per-film + mean."""
    rows = []
    for ablation, mode_dict in results.items():
        for mode, fold_list in mode_dict.items():
            row = {"ablation": ablation, "eval_mode": mode}
            precisions = []
            recalls    = []
            for fold in fold_list:
                film = fold["test_film"]
                row[f"{film}_dt_prec"] = fold["dont_trust_precision"]
                row[f"{film}_dt_rec"]  = fold["dont_trust_recall"]
                precisions.append(fold["dont_trust_precision"])
                recalls.append(fold["dont_trust_recall"])
            if precisions:
                row["mean_dt_prec"] = float(np.mean(precisions))
                row["std_dt_prec"]  = float(np.std(precisions))
                row["mean_dt_rec"]  = float(np.mean(recalls))
                row["std_dt_rec"]   = float(np.std(recalls))
            rows.append(row)
    return pd.DataFrame(rows)


def print_master_table(results):
    """The single most important output: ablation x eval_mode summary."""
    print("\n" + "=" * 72)
    print("MASTER SUMMARY: dont_trust precision (mean +/- std across folds)")
    print("=" * 72)
    header = f"{'ablation':16s} {'eval_mode':12s} {'dt_precision':>16s} {'dt_recall':>16s}"
    print(header)
    print("-" * len(header))
    for ablation in ABLATIONS:
        for mode in ("filtered", "unfiltered"):
            folds = results[ablation][mode]
            if not folds:
                print(f"{ablation:16s} {mode:12s} {'(no folds)':>16s}")
                continue
            ps = [f["dont_trust_precision"] for f in folds]
            rs = [f["dont_trust_recall"]    for f in folds]
            print(f"{ablation:16s} {mode:12s} "
                  f"{np.mean(ps):>7.3f} +/- {np.std(ps):>4.3f} "
                  f"{np.mean(rs):>7.3f} +/- {np.std(rs):>4.3f}")

    # Per-fold breakdown for the full classifier (most thesis-relevant)
    print("\n" + "=" * 72)
    print("PER-FOLD BREAKDOWN: 'full' ablation, filtered evaluation")
    print("=" * 72)
    for fold in results["full"]["filtered"]:
        cm = fold["confusion_matrix"]
        print(f"  {fold['test_film']:20s} "
              f"dt_prec={fold['dont_trust_precision']:.3f}  "
              f"dt_rec={fold['dont_trust_recall']:.3f}  "
              f"trust_prec={fold['trust_precision']:.3f}  "
              f"trust_rec={fold['trust_recall']:.3f}  "
              f"n_dt_test={fold['dont_trust_support']}  "
              f"cm={cm}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(csv_path, summary_csv_path):
    df = load_and_validate(csv_path)
    results = run_lofo_cv(df)
    print_master_table(results)

    summary = summarize_results(results)
    summary.to_csv(summary_csv_path, index=False)
    print(f"\n[output] Wrote summary to: {summary_csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default = '../Feature_Engineering/Long Data.csv', required=True,
                    help="Path to Long Data.csv produced by feature_engineering_csv.py" )
    ap.add_argument("--out", default="lofo_summary.csv",
                    help="Path for summary CSV output (default: lofo_summary.csv)")
    args = ap.parse_args()
    main(args.csv, args.out)