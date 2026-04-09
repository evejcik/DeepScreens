import numpy as np
import pandas as pd
import argparse
from scipy.stats import beta

from beta_MAP import jitter, moments, beta_fit, posterior

import joblib, pathlib

from isotonic_regression import fit_global_isotonic
##to be used when a joint has very few visible samples (i.e., Ramona right ankle only has 9 visible samples, I'm sure some down the road will have 0 visible samples for ankles, feet, etc.)''

#first need to estimate the global prior - all confidence scores from csv, for all visibility classes
def global_prior(df):
    #Fraction of rows that are labelled visible (visibility_category == 1)
    n_vis = (df["visibility_category"] == 1).sum() #or could be np.sum(df["visibility_category"] == 1)
    return float(n_vis / len(df))


def posterior_from_isotonic(scores, iso):
    return iso.predict(np.atleast_1d(scores))

def global_beta(scores):
    # scores = df['mmpose_confidence']
    jittered_scores = jitter(scores)
    mu, var, k, alpha0, beta0 = moments(jittered_scores)

    if np.isnan(alpha0) or np.isnan(beta0):
        print(f"  → MoM failed on {len(scores)} samples; returning (2, 2)")
        return 2.0, 2.0  # ← fallback

    global_a, global_b, _ = beta_fit(jittered_scores, alpha_prior=2.0, beta_prior=2.0)

    if np.isnan(global_a) or np.isnan(global_b):
        print(f"  → Optimizer failed; using MoM: α={alpha0:.2f}, β={beta0:.2f}")
        return float(alpha0), float(beta0)  # ← fallback to MoM

    return global_a, global_b 

#optionally refine those a_0 and b_0 values with an MAP optimisation using the same weak prior (2,2) as in beta MAP.py. the result a_0, b_0 is the global beta.
#because we are using ALL data, the global variance is never zero, so the method of moments (mom) estimate is stable.

#once we have the global prior, we can pass the global parameters of this distribution into the loss function for a given point, especially if the number of samples is low for that specific one.

def main(data):
    df = pd.read_csv(data)
    # print(df.columns)
    scores = df['mmpose_confidence']
    global_a, global_b = global_beta(scores) 

    pi_global = global_prior(df)
    iso_glonal = fit_global_isotonic(df)

    scores_vis = df[df['visibility_category'] == 1]['mmpose_confidence'].values
    # print(np.sum(scores_vis))
    a_vis_global, b_vis_global = global_beta(scores_vis)

    scores_not_vis = df[df['visibility_category'] != 1]['mmpose_confidence'].values
    a_not_vis_global, b_not_vis_global = global_beta(scores_not_vis)

    # ← fallback if NaN
    if np.isnan(a_vis_global):
        print("Warning: Global visible Beta failed; falling back to (2, 2)")
        a_vis_global, b_vis_global = 2.0, 2.0
    if np.isnan(a_not_vis_global):
        print("Warning: Global not-visible Beta failed; falling back to (2, 2)")
        a_not_vis_global, b_not_vis_global = 2.0, 2.0

    print(f"\n=== Global hyper-parameters ===")
    print(f"Global visible Beta   : a={a_vis_global:.2f}, β={b_vis_global:.2f}")
    print(f"Global not-vis Beta   : a={a_not_vis_global:.2f}, β={b_not_vis_global:.2f}")
    

    results = {}
    joints = df['joint_name'].unique()

    for joint in joints:
        results[joint] = {}

        for vis_label, vis_value, prior_a, prior_b in [
                ("visible",     [1], a_vis_global, b_vis_global),          # <-- wrap 1 in a list
                ("not_visible", [2, 3, 4], a_not_vis_global, b_not_vis_global)]:   # already a list
            mask = (df["joint_name"] == joint) & df["visibility_category"].isin(vis_value)

            scores = df.loc[mask, "mmpose_confidence"].values
            
            if scores.size == 0:
                # No data for this joint/visibility – we store NaNs so downstream code can detect it.
                results[joint][vis_label] = {"a": np.nan, "b": np.nan,
                                             "n": 0, "success": False}
                continue

            if scores.size < 10:  # ultra‑low sample count
                print(f"DEBUG: {joint} {vis_label} n={scores.size} → using global prior")
                # Skip the per‑joint fit; use the global prior as the final estimate
                alpha_opt, beta_opt = (a_vis_global, b_vis_global) if vis_label == "visible" \
                                    else (a_not_vis_global, b_not_vis_global)
                ok = False
            else:
                alpha_opt, beta_opt, ok = beta_fit(
                    scores,
                    alpha_prior = prior_a,
                    beta_prior = prior_b)

            results[joint][vis_label] = {
                "a": alpha_opt,
                "b": beta_opt,
                "n": int(scores.size),
                "success": ok
            }

    print("\n=== Beta MAP fit summary ===")
    for joint, d in results.items():
        vis = d["visible"]
        not_vis = d["not_visible"]
        print(f"{joint:>12} | "
              f"vis n={vis['n']:4d}  α={vis['a']:.2f} β={vis['b']:.2f} "
              f"| not_vis n={not_vis['n']:4d}  α={not_vis['a']:.2f} β={not_vis['b']:.2f}")
    print("\n=== Example of the full posterior (global) ===")


    for c in [0.95, 0.98, 0.99, 0.999]:
        p = posterior_global(
                c,
                pi=pi_global,
                a_vis=a_vis_global,
                b_vis=b_vis_global,
                a_not_vis=a_not_vis_global,
                b_not_vis=b_not_vis_global)
        print(f"c={c:.3f} → P(visible|c) = {p:.3f}")


def posterior_global(scores, pi, a_vis, b_vis, a_not_vis, b_not_vis): #p(visible | scores) means we also need to miltiple beta likelihood by pi (global prior) 
    #and complementary (1-pi) for inivisible class
    pdf_vis = beta.pdf(scores, a_vis, b_vis)
    pdf_not_vis = beta.pdf(scores, a_not_vis, b_not_vis)
    eps = float(1e-12)
    num = pi * pdf_vis
    denom = num + (1.0 - pi) * pdf_not_vis + eps

    return num/denom

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")

    args = ap.parse_args()
    main(data = args.data)

