"""
error_analysis.py
-----------------
Diagnostic post-hoc analysis of the LOFO-CV `full` classifier on a chosen
test film (default: both Tron folds combined). The goal is to answer two
specific questions:

  Q1: When the classifier predicts dont_trust but ground truth is trust
      (false positives / over-flagging), what do those rows have in
      common compared to correctly-classified trust rows?

  Q2: When the classifier predicts trust but ground truth is dont_trust
      (false negatives / missed flags), what do those rows have in common
      compared to correctly-classified dont_trust rows?

These two questions decompose the classifier's failure modes into the two
ways the precision number gets hurt. Q1 errors directly hurt precision
(over-flagging). Q2 errors hurt recall but also tell you where the classifier
is missing genuine reliability problems.

This script does NOT optimize the classifier. It diagnoses it. The output
is a set of feature-level summary tables and per-joint error rates that
you read by eye to decide what to do next.

Usage
-----
python error_analysis.py --csv "Long Data.csv"
    Default: analyze Tron_2059_2148 + Tron_3067_3132 combined.

python error_analysis.py --csv "Long Data.csv" --test_films Moonlight_1_1529
    Single film analysis.

python error_analysis.py --csv "Long Data.csv" --test_films Tron_2059_2148 Tron_3067_3132
    Multiple films analyzed jointly.

What it prints
--------------
For each requested test film (or combination):
  1. Confusion matrix and basic precision/recall numbers
  2. False-positive analysis: feature distributions for FPs vs correct trusts
  3. False-negative analysis: feature distributions for FNs vs correct
     dont_trusts
  4. Per-joint error rates (which joint identities does the classifier
     misclassify most often?)
  5. If reason_for_distrust is in the CSV, error breakdown by annotated
     reason (mask flipping vs occlusion vs other)
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.3f}".format)


# ---------------------------------------------------------------------------
# Configuration -- must match classifier.py for results to be comparable
# ---------------------------------------------------------------------------

ALL_FILMS = [
    "Moonlight_1_1529",
    "Ramona_1_1639",
    "Tron_2059_2148",
    "Tron_3067_3132",
    "Psycho_319_1411",
    "Psycho_319_2006",
]

# Same feature set as the `full` ablation in classifier.py (no leaking
# label-derived features).
FEATURES = [
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
]

TARGET = "reliability_category_int"
DONT_TRUST = 2
TRUST = 0


# ---------------------------------------------------------------------------
# Data loading (matches classifier.py)
# ---------------------------------------------------------------------------

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    for col in FEATURES:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(-1)
    return df


# ---------------------------------------------------------------------------
# Run classifier on the held-out fold
# ---------------------------------------------------------------------------

def train_and_predict(train_df, test_df):
    """Match classifier.py's `full` LightGBM exactly."""
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    clf = lgb.LGBMClassifier(
        class_weight="balanced",
        random_state=0,
        n_estimators=200,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, list(clf.classes_).index(DONT_TRUST)]
    return y_pred, y_proba, clf


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def confusion_summary(y_true, y_pred):
    tp = ((y_true == DONT_TRUST) & (y_pred == DONT_TRUST)).sum()
    fp = ((y_true == TRUST)      & (y_pred == DONT_TRUST)).sum()
    fn = ((y_true == DONT_TRUST) & (y_pred == TRUST)).sum()
    tn = ((y_true == TRUST)      & (y_pred == TRUST)).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "dont_trust_precision": float(precision),
        "dont_trust_recall":    float(recall),
    }


def feature_distribution_table(test_df, group_mask, comparison_mask, label):
    """
    For each FEATURE, compute mean/median/std on the error group and the
    comparison group, plus the standardized mean difference (Cohen's d-like).
    Larger absolute values mean the feature is more discriminative between
    the two groups.
    """
    rows = []
    for feat in FEATURES:
        a = test_df.loc[group_mask, feat].dropna().to_numpy()
        b = test_df.loc[comparison_mask, feat].dropna().to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        ma, mb = a.mean(), b.mean()
        sa, sb = a.std(),  b.std()
        # Pooled std for standardized difference
        pooled = np.sqrt((sa**2 + sb**2) / 2)
        diff = (ma - mb) / pooled if pooled > 0 else 0.0
        rows.append({
            "feature":         feat,
            f"{label}_mean":   ma,
            "comparison_mean": mb,
            f"{label}_std":    sa,
            "std_diff":        diff,
        })
    out = pd.DataFrame(rows)
    out = out.reindex(out["std_diff"].abs().sort_values(ascending=False).index)
    return out


def per_joint_error_rates(test_df, y_true, y_pred):
    """
    Per joint identity, compute count and rate of FPs and FNs. Sorted by
    total error count.
    """
    df = test_df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["is_fp"] = ((df["y_true"] == TRUST)      & (df["y_pred"] == DONT_TRUST)).astype(int)
    df["is_fn"] = ((df["y_true"] == DONT_TRUST) & (df["y_pred"] == TRUST)).astype(int)
    df["is_correct"] = (df["y_true"] == df["y_pred"]).astype(int)

    g = df.groupby("joint_name").agg(
        n_total=("y_true", "size"),
        n_trust=("y_true", lambda s: int((s == TRUST).sum())),
        n_dont_trust=("y_true", lambda s: int((s == DONT_TRUST).sum())),
        n_fp=("is_fp", "sum"),
        n_fn=("is_fn", "sum"),
        accuracy=("is_correct", "mean"),
    )
    g["fp_rate_of_trusts"] = g.apply(
        lambda r: r["n_fp"] / r["n_trust"] if r["n_trust"] else 0.0, axis=1
    )
    g["fn_rate_of_donts"] = g.apply(
        lambda r: r["n_fn"] / r["n_dont_trust"] if r["n_dont_trust"] else 0.0, axis=1
    )
    g["total_errors"] = g["n_fp"] + g["n_fn"]
    return g.sort_values("total_errors", ascending=False)


def reason_breakdown(test_df, y_true, y_pred):
    """If reason_for_distrust is present, break down FNs by annotated reason."""
    if "reason_for_distrust" not in test_df.columns:
        return None
    df = test_df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    # Only dont_trust rows have a reason; subset to those.
    dt = df[df["y_true"] == DONT_TRUST].copy()
    dt["correctly_flagged"] = (dt["y_pred"] == DONT_TRUST).astype(int)

    # Some CSVs use NaN, -1, or empty string. Normalize to a single bucket.
    dt["reason_norm"] = dt["reason_for_distrust"].fillna("(none)")
    dt.loc[dt["reason_norm"] == -1, "reason_norm"] = "(none)"
    dt.loc[dt["reason_norm"] == "-1", "reason_norm"] = "(none)"

    g = dt.groupby("reason_norm").agg(
        n_dont_trust=("y_true", "size"),
        n_caught=("correctly_flagged", "sum"),
    )
    g["recall_for_reason"] = g["n_caught"] / g["n_dont_trust"]
    return g.sort_values("n_dont_trust", ascending=False)


def feature_importance(clf, top_n=12):
    fi = pd.DataFrame({
        "feature":    FEATURES,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)
    return fi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_one(df, test_films):
    test_label = "+".join(test_films)
    print("\n" + "=" * 78)
    print(f"ERROR ANALYSIS  -- test fold: {test_label}")
    print("=" * 78)

    train_df = df[~df["film"].isin(test_films)].reset_index(drop=True)
    test_df  = df[ df["film"].isin(test_films)].reset_index(drop=True)
    if len(test_df) == 0:
        print(f"No test rows for {test_label}. Skipping.")
        return

    train_joints = set(train_df["joint_name"].unique())
    test_filtered_mask = test_df["joint_name"].isin(train_joints)
    test_df = test_df[test_filtered_mask].reset_index(drop=True)
    print(f"Train rows: {len(train_df):>6d}  (films: {sorted(train_df['film'].unique())})")
    print(f"Test rows:  {len(test_df):>6d}  (joints filtered to those seen in training)")

    y_pred, y_proba, clf = train_and_predict(train_df, test_df)
    y_true = test_df[TARGET].values

    # 1. Confusion summary
    cs = confusion_summary(y_true, y_pred)
    print(f"\n  Confusion: TP={cs['tp']}  FP={cs['fp']}  FN={cs['fn']}  TN={cs['tn']}")
    print(f"  dont_trust precision: {cs['dont_trust_precision']:.3f}")
    print(f"  dont_trust recall:    {cs['dont_trust_recall']:.3f}")
    print(f"  Probability of dont_trust (test predictions):")
    print(f"    median={np.median(y_proba):.3f}   "
          f"mean={np.mean(y_proba):.3f}   "
          f"std={np.std(y_proba):.3f}")

    # 2. Feature importance
    print("\n  Top feature importances (LightGBM):")
    fi = feature_importance(clf)
    print(fi.to_string(index=False))

    # 3. False-positive analysis
    fp_mask = (y_true == TRUST) & (y_pred == DONT_TRUST)
    correct_trust_mask = (y_true == TRUST) & (y_pred == TRUST)
    print(f"\n  FALSE POSITIVES (over-flagged): n={int(fp_mask.sum())} "
          f"vs correctly-trusted n={int(correct_trust_mask.sum())}")
    if fp_mask.sum() > 0 and correct_trust_mask.sum() > 0:
        fp_table = feature_distribution_table(
            test_df, fp_mask, correct_trust_mask, "fp")
        print("  Features ranked by std-diff (large |std_diff| = discriminative):")
        print(fp_table.to_string(index=False))
    else:
        print("  Not enough rows in one of the groups to compute distributions.")

    # 4. False-negative analysis
    fn_mask = (y_true == DONT_TRUST) & (y_pred == TRUST)
    correct_dt_mask = (y_true == DONT_TRUST) & (y_pred == DONT_TRUST)
    print(f"\n  FALSE NEGATIVES (missed flags): n={int(fn_mask.sum())} "
          f"vs correctly-distrusted n={int(correct_dt_mask.sum())}")
    if fn_mask.sum() > 0 and correct_dt_mask.sum() > 0:
        fn_table = feature_distribution_table(
            test_df, fn_mask, correct_dt_mask, "fn")
        print("  Features ranked by std-diff:")
        print(fn_table.to_string(index=False))
    else:
        print("  Not enough rows in one of the groups to compute distributions.")

    # 5. Per-joint error breakdown
    print("\n  Per-joint error rates:")
    pj = per_joint_error_rates(test_df, y_true, y_pred)
    print(pj.to_string())

    # 6. Reason-for-distrust breakdown if available
    rb = reason_breakdown(test_df, y_true, y_pred)
    if rb is not None and len(rb) > 0:
        print("\n  Recall by annotated reason_for_distrust (FN-relevant):")
        print(rb.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="Path to Long Data.csv from feature_engineering_csv.py")
    ap.add_argument("--test_films", nargs="+",
                    default=["Tron_2059_2148", "Tron_3067_3132"],
                    help="One or more film names to analyze together as a "
                         "single LOFO fold. Default: both Tron segments.")
    ap.add_argument("--also_separately", action="store_true",
                    help="In addition to the combined analysis, run an "
                         "additional analysis for each test film alone.")
    args = ap.parse_args()

    df = load_and_clean(args.csv)
    print(f"Loaded {len(df)} rows. Films present: {sorted(df['film'].unique())}")

    analyze_one(df, args.test_films)
    if args.also_separately and len(args.test_films) > 1:
        for f in args.test_films:
            analyze_one(df, [f])


if __name__ == "__main__":
    main()