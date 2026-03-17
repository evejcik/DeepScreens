#look at metrics per csv

import numpy as np
import pandas as pd
import argparse

def main(df):
    df = pd.read_csv(df)
    scores = df['mmpose_confidence']

    scores_mean = np.mean(scores)
    scores_std = np.std(scores)

    percentage_groups = {}

    total = len(df)
    visible_count = (df['visibility_category'] == 1).sum()
    not_visible_count = ((df['visibility_category'] == 2) | (df['visibility_category'] == 3)).sum()
    pct_visible = round(100 * visible_count/total, 2)
    pct_not_visible = round(100 * not_visible_count/total, 2)

    joint_stats = (
        df.groupby("joint_name")["mmpose_confidence"]
          .agg(["mean", "std", "count"])
          .rename(columns={"count": "n_samples"})
    )

    report = {
        "total_frames"          : total,
        "visible_count"         : visible_count,
        "not_visible_count"     : not_visible_count,
        "visible_percent"       : pct_visible,
        "not_visible_percent"   : pct_not_visible,
        "confidence_mean"       : scores_mean,
        "confidence_std"        : scores_std,
        "joint_stats"           : joint_stats,
    }

    return report


if __name__== "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--df")
    args = ap.parse_args()

    print(main(args.df))




