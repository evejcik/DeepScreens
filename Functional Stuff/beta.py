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

from sklearn.linear_model import LogisticRegression

def relative_confidences(scores, confidence_method):
    if confidence_method == 'transform':
        #widen spread of confidence scores
        scores = np.log(scores + 1e-6)
    elif confidence_method == 'mse': #use MSE instead of MLE
    elif confidence_method == 'log_regression': #need to use better model to fit tiny variance, so model reverse -> p(visible | confidence)
    elif confidence_method == 'relative_confidence': #compare confidence score to mean body confidence score

    return scores


def beta_per_joint(df, joint, confidence_method):
    #input: dataset and 1 joint name
    #output: dictionary with fitted parameters

    joint_data = df[df['joint_name'] == joint].copy()
    joint_data['is_visible'] = (joint_data['visibility_category'] == 1).astype(int)

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
        scores = transform_confidences(scores, confidence_method)

        ##GUARDRAILS:
        #sometimes, a joint never has any visibility throughout the whole video.
        #when this happens, scores = [], empty. this will cause errors later down the road, i.e., when calculating mean, etc.
        #in order to accomodate this...

        ##SIMILARLY, if there is near-zero variance in our beta distribution...
        #then the solver tries very big betas (b) and very big alphas (a)
        #so... 
        # Numerically this means that: 
        # gradients become unstable
        # the likelihood surface becomes flat/ill-conditioned
        # solver stops: “not making good progress”
        #this means that the "best fit" distribution would look like a vertical spike at 1

        # if len(scores) < 2:
        #     print(f"{joint} ({visibility}): skipped (n={len(scores)})")
        #     continue

        # if np.var(scores) < 1e-6:
        #     print(f"{joint} ({visibility}): skipped (near-constant)")
        #     continue


        a, b, loc, scale = beta.fit(scores, floc = floc, fscale = fscale)

        results[visibility] = {'a': a, 'b': b, 'samples': len(scores), 'mean': scores.mean()}
        print(f"{joint} ({visibility}): a = {a:.2f}, b = {b:.2f}, n = {len(scores)}, mean = {scores.mean():.3f}")

    return results

def fit_all_joints(df):
    betas = {}
    for joint in pd.Series(df['joint_name']).unique():
        betas[joint] = beta_per_joint(df, joint)
        
    return betas

def save_beta_parameters(betas):
    with open('fitted_betas.pkl', 'wb') as f:
        pickle.dump(betas, f)
    print(f"Saved fitted Beta parameters to fitted_betas.pkl")

def main(df, confidence_method):
    save_beta_parameters(fit_all_joints(pd.read_csv(df), confidence_method))
    print("done!")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--df")
    ap.add_argument("--confidence_method")

    args = ap.parse_args()
    main(args.df, args.confidence_method)

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
    
#     ap.add_argument("--json", required = True)
#     ap.add_argument("--mp4", required = True)
#     ap.add_argument("--end", type = int)
#     ap.add_argument("--create_new_df", type = int)
#     ap.add_argument("--start", type = int)
#     ap.add_argument("--video_nobbox", default = None)

#     args = ap.parse_args()
#     main(args.mp4, args.json, args.start, args.end, args.create_new_df, args.video_nobbox)


