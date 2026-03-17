#laying down groundwork for simple beta distribution
#we are looking at: p(joint is visible | visibility label, confidence score, any other features from data)
#confidences are bounded 0-1

#we can interpret the beta distribution's 2 parameters cleanly: the number of successes vs the number of failures -> the number of times the joint is visible vs not

from scipy.stats import beta
import argparse
import numpy as np
import pandas as pd
import pickle #to save parameters for later
from pathlib import Path


def beta_per_joint(df, joint):
    #input: dataset and 1 joint name
    #output: dictionary with fitted parameters

    joint_data = df[df['joint_name'] == joint].copy()
    joint_data['is_visible'] = (joint_data['visibility'] == 1).astype(int)

    results = {}

    #need to set loc and scale for scipy stats to bound problem
    #scipy beta uses MLE to calculate a and b for beta distribution
    floc = 0
    fscale = 1 #this fits the distribution to standard [0,1]
    #this works since we are working with probabilities (the confidences) bounded by 0 and 1, so we don't have to bring in unbounded MSE calculations for now
    #so now only alpha and beta are estimated
    for val in [0,1]: #p(joint is visible | confidence scores)
        visibility = 'visible' if val == 1 else 'not_visible'
        scores = joint_data[joint_data['is_visible'] == val]['mmpose_confidence'].values
        scores = np.clip(scores, 0.001, 0.999)
        alpha, beta, loc, scale = beta.fit(scores, floc = floc, fscale = fscale)

        results[visibility] = {'a': alpha, 'b': beta, 'samples': len(scores), 'mean': scores.mean()}
        print(f"{joint_name} ({vis_label}): a = {a:.2f}, b = {b:.2f}, n = {len(scores)}, mean = {scores.mean():.3f}")

    return results

def fit_all_joints(df):
    betas = {}
    for joint in df['joint_name'].unique():
        betas[joint] = beta_per_joint(df, joint)
    return betas

def save_beta_parameters(betas):
    with open('fitted_betas.pkl', 'wb') as f:
        pickle.dump(betas, f)
    print(f"Saved fitted Beta parameters to fitted_betas.pkl")


def main(df):
    save_beta_parameters(fit_all_joints(df))

if __name__ == "main":
    ap.argParse.ArgumentParser()
    ap.add_argument('df')

    ap.parse_args()
    main(df)



