"""
per_tier_evaluation.py
----------------------
Reports per-tier dont_trust precision and recall for the LOFO-CV `full`
classifier. Tiers are defined by their downstream impact on animation quality:

  Tier 1 (skeleton anchors):  hips, shoulders
                              wrong flag here corrupts root, spine, thorax
                              and propagates to the entire skeleton

  Tier 2 (limb extremities):  knees, ankles, elbows, wrists
                              wrong flag affects one limb segment

  Tier 3 (head/derived):      head, neck_base, spine, root, thorax
                              wrong flag affects only the joint itself
                              (or is propagated from sources for derived joints)

For an animation pipeline, the cost of a mistake scales with tier:
Tier 1 errors corrupt whole-body motion; Tier 3 errors are localized.

This script extends classifier.py's evaluation by computing precision and
recall per tier per fold, then aggregating.

Usage
-----
python per_tier_evaluation.py --csv "Long Data.csv"

Optional: --exclude_films Tron_2059_2148 Tron_3067_3132
   Drops listed films from BOTH training and test sets. Use this to compute
   archival-only headline numbers excluding Tron.
"""

import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.3f}".format)


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

# Same feature set as classifier.py's `full` ablation.
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


# Tiers per project notes:
#   Tier 1: skeleton anchors. Wrong here propagates to root, spine, thorax
#           (which derive from hips/shoulders) and corrupts whole-body motion.
#   Tier 2: limb extremities. Wrong here affects one limb segment.
#   Tier 3: head and derived. Localized impact (or already-localized for derived).
JOINT_TIERS = {
    # Tier 1 -- skeleton anchors
    "left_hip":       1,
    "right_hip":      1,
    "left_shoulder":  1,
    "right_shoulder": 1,

    # Tier 2 -- limb extremities
    "left_knee":      2,
    "right_knee":     2,
    "left_ankle":     2,
    "right_ankle":    2,
    "left_elbow":     2,
    "right_elbow":    2,
    "left_wrist":     2,
    "right_wrist":    2,

    # Tier 3 -- head/derived (affect single visual joint or are computed from sources)
    "head":           3,
    "neck_base":      3,
    "spine":          3,
    "thorax":         3,
    "root":           3,
}

TIER_LABELS = {
    1: "Tier 1 (anchors)",
    2: "Tier 2 (extremities)",
    3: "Tier 3 (head/derived)",
}


# ---------------------------------------------------------------------------
# Data loading (matches classifier.py)
# ---------------------------------------------------------------------------

def load_and_clean(csv_path, exclude_films):
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    for col in FEATURES:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(-1)

    if exclude_films:
        before = len(df)
        df = df[~df["film"].isin(exclude_films)].reset_index(drop=True)
        print(f"[data] Excluded films {exclude_films}: dropped "
              f"{before - len(df)} rows. Remaining: {len(df)}")

    return df


# ---------------------------------------------------------------------------
# Train and predict (LightGBM full ablation)
# ---------------------------------------------------------------------------

def train_and_predict(train_df, test_df):
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    clf = lgb.LGBMClassifier(
        class_weight="balanced",
        random_state=0,
        n_estimators=200,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


# ---------------------------------------------------------------------------
# Per-tier metrics
# ---------------------------------------------------------------------------

def compute_per_tier_metrics(test_df, y_pred):
    """
    For each tier in {1, 2, 3}, compute dont_trust precision/recall
    on test rows whose joint_name maps to that tier.
    """
    df = test_df.copy()
    df["y_true"] = test_df[TARGET].values
    df["y_pred"] = y_pred
    df["tier"]   = df["joint_name"].map(JOINT_TIERS)

    out = {}
    for tier in [1, 2, 3]:
        mask = df["tier"] == tier
        sub = df[mask]
        if len(sub) == 0 or sub["y_true"].nunique() < 2:
            out[tier] = {
                "n_total":        int(len(sub)),
                "n_dont_trust":   int((sub["y_true"] == DONT_TRUST).sum()),
                "dt_precision":   None,
                "dt_recall":      None,
            }
            continue

        labels = [TRUST, DONT_TRUST]
        p, r, _, sup = precision_recall_fscore_support(
            sub["y_true"], sub["y_pred"], labels=labels, zero_division=0
        )
        out[tier] = {
            "n_total":      int(len(sub)),
            "n_dont_trust": int(sup[1]),
            "dt_precision": float(p[1]),
            "dt_recall":    float(r[1]),
        }
    return out


# ---------------------------------------------------------------------------
# LOFO-CV
# ---------------------------------------------------------------------------

def run_lofo_cv(df):
    """
    Run LOFO-CV. For each fold, hold out one film, train on the rest,
    record per-tier metrics on the test fold (filtered to seen joints).
    Returns dict[tier] -> list of per-fold dicts.
    """
    test_films = sorted(df["film"].unique())
    print(f"\nRunning LOFO-CV across {len(test_films)} films:")
    for f in test_films:
        print(f"  - {f}")

    per_fold = []

    for test_film in test_films:
        train_df = df[df["film"] != test_film].reset_index(drop=True)
        test_df  = df[df["film"] == test_film].reset_index(drop=True)
        train_joints = set(train_df["joint_name"].unique())
        test_filtered_df = test_df[test_df["joint_name"].isin(train_joints)].reset_index(drop=True)

        if len(test_filtered_df) == 0:
            print(f"\n[fold] {test_film}: SKIP (no joints overlap with training)")
            continue

        y_pred = train_and_predict(train_df, test_filtered_df)
        tier_metrics = compute_per_tier_metrics(test_filtered_df, y_pred)
        tier_metrics["test_film"] = test_film
        per_fold.append(tier_metrics)

        print(f"\n[fold] {test_film}")
        for tier in [1, 2, 3]:
            m = tier_metrics[tier]
            if m["dt_precision"] is None:
                print(f"  {TIER_LABELS[tier]:25s}  n={m['n_total']:>5d}  "
                      f"n_dt={m['n_dont_trust']:>4d}  -- (no class balance)")
            else:
                print(f"  {TIER_LABELS[tier]:25s}  n={m['n_total']:>5d}  "
                      f"n_dt={m['n_dont_trust']:>4d}  "
                      f"dt_prec={m['dt_precision']:.3f}  "
                      f"dt_rec={m['dt_recall']:.3f}")

    return per_fold


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def aggregate_and_report(per_fold):
    """Print master tables: per-tier mean/std across folds, weighted and unweighted."""

    print("\n" + "=" * 72)
    print("PER-TIER MASTER SUMMARY (across all LOFO folds)")
    print("=" * 72)

    # Unweighted: each fold contributes equally
    print(f"\n{'Tier':25s}  {'Unweighted mean (std)':30s}  {'Weighted by n_dt':30s}")
    print("-" * 90)
    for tier in [1, 2, 3]:
        precs = []
        recs  = []
        n_dts = []
        for fold in per_fold:
            m = fold[tier]
            if m["dt_precision"] is not None and m["n_dont_trust"] > 0:
                precs.append(m["dt_precision"])
                recs.append(m["dt_recall"])
                n_dts.append(m["n_dont_trust"])

        if not precs:
            print(f"{TIER_LABELS[tier]:25s}  (no valid folds)")
            continue

        precs = np.array(precs)
        recs  = np.array(recs)
        n_dts = np.array(n_dts)

        unw_p = precs.mean()
        unw_r = recs.mean()
        unw_p_std = precs.std()
        unw_r_std = recs.std()

        # Weight each fold's precision by its n_dont_trust support
        w = n_dts / n_dts.sum()
        wgt_p = (precs * w).sum()
        wgt_r = (recs * w).sum()

        unw_str = f"prec={unw_p:.3f}+/-{unw_p_std:.3f}  rec={unw_r:.3f}+/-{unw_r_std:.3f}"
        wgt_str = f"prec={wgt_p:.3f}             rec={wgt_r:.3f}"
        print(f"{TIER_LABELS[tier]:25s}  {unw_str:30s}  {wgt_str:30s}")

    # Per-fold per-tier table
    print("\n" + "=" * 72)
    print("PER-FOLD PER-TIER BREAKDOWN")
    print("=" * 72)
    header = f"{'film':22s} {'tier':8s} {'n':>5s} {'n_dt':>5s} {'dt_prec':>9s} {'dt_rec':>9s}"
    print(header)
    print("-" * len(header))
    for fold in per_fold:
        for tier in [1, 2, 3]:
            m = fold[tier]
            prec_str = f"{m['dt_precision']:.3f}" if m["dt_precision"] is not None else "  --  "
            rec_str  = f"{m['dt_recall']:.3f}"    if m["dt_recall"]    is not None else "  --  "
            print(f"{fold['test_film']:22s} {tier:>4d}    "
                  f"{m['n_total']:>5d} {m['n_dont_trust']:>5d} "
                  f"{prec_str:>9s} {rec_str:>9s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="Path to Long Data.csv from feature_engineering_csv.py")
    ap.add_argument("--exclude_films", nargs="*", default=[],
                    help="Films to drop from BOTH training and test sets. "
                         "Useful for excluding Tron when reporting "
                         "archival-only metrics.")
    args = ap.parse_args()

    df = load_and_clean(args.csv, args.exclude_films)
    print(f"\nFilms in data: {sorted(df['film'].unique())}")
    print(f"Joint name distribution:")
    print(df["joint_name"].value_counts().to_string())
    print(f"\nTier distribution (annotated rows):")
    df_tier = df.copy()
    df_tier["tier"] = df_tier["joint_name"].map(JOINT_TIERS)
    print(df_tier.groupby("tier")[TARGET].value_counts().to_string())

    per_fold = run_lofo_cv(df)
    aggregate_and_report(per_fold)


if __name__ == "__main__":
    main()