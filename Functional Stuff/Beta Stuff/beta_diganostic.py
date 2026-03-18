import pandas as pd
import numpy as np
import argparse
from scipy.special import betaln   # log of the beta function B(α,β)


def moments(scores): #quick, closed‑form way to pick α and β just from the sample mean and sample variance of the scores.
    #uses: as an initial guess for a more refined optimizer (MLE or MAP),
    # as a fallback when the optimizer fails, and
    # as a simple sanity check (does the fitted distribution even look plausible?).

    sample_mean = np.mean(scores) #first moment
    sample_var = np.var(scores) #second spread out moment
    k = sample_mean * (1 - sample_mean) / sample_var

    alpha_mom = sample_mean * k #mom = Method of Moments
    beta_mom = (1 - sample_mean) * k
    return sample_mean, sample_var, k, alpha_mom, beta_mom



##negative log posterior J(alpha, beta)
def neg_log_posterior(alpha, beta, scores, eps: float = 1e-6, alpha_prior = 2.0, beta_prior = 2.0): 
    #MLE out of the question, confidence scores are all wayyyyy too close to 1 so the optimizer is really struggling to find a good alpha & beta
    #so we choose MAP with weak Beta (2,2) prior
    ##FOR OPTIMIZATION'S SAKE: consider placing a different distribution over alpha and beta parameters 
    # i.e., a gamma or hierarchical bayes distrbution
    #as opposed to the initial exponential distribution

    c = scores
    c = jitter(scores, eps)
    N = len(scores)

    #(alpha - 1)
    alpha_1 = (alpha - 1.0)
    #sigma log(c_i) 
    sigma_log_ci = np.sum(np.log(c))
    #beta -1
    beta_1 = (beta - 1.0)

    sigma_log_1_ci = np.sum(np.log(1.0 - c))
    n_log_beta = N * np.log(betaln(alpha, beta))

    negative_log_likelihood = alpha_1 * sigma_log_ci + beta_1 * sigma_log_1_ci - n_log_beta

    ##we get this part by taking the log of the exponential distribution on alpha and beta -> if we change this, we change it here
    log_prior = - (alpha - alpha_prior) - (beta - beta_prior) #right now, set to 2 and 2 to keep prior weak 

    final = negative_log_likelihood - log_prior

    return float(final)

## Logit Gaussian

## Jitter and clip -> alternatives:  Hierarchical / Empirical‑Bayes pooling & Isotonic regression (non‑parametric)

def jitter(scores, epsilon = float(1e-6)): #data now has non zero variance. A zero variance, for a beta distribution, means that when the optimizer
    #is trying to fit the best alpha and beta to the distribution, beta goes to 0 and alpha goes to infinity -> not good for trying to find a reasonal 
    #distribution of probabilities for the confidence scores
    scores = np.clip(epsilon, 1 - epsilon)
    jittered_scores = scores + epsilon * np.random.default_rng().normal()

    clipped_jittered_scores = np.clip(jittered_scores, epsilon, 1 - epsilon)

    return clipped_jittered_scores

## Hierarchical / Empirical‑Bayes pooling

## Isotonic regression (non‑parametric)

## Simple Binning
def main(data):
    joints = data["joint_name"].unique()
    for joint in joints:
        joint_data = data.loc[[joint]]['mmpose_confidence']
        # scores = joint_data['mmpose_confidence']
        sample_mean, sample_var, k, alpha_mom, beta_mom = moments(joint_data)
        neg_log_posterior(alpha = alpha_mom, beta = beta_mom, scores = joint_data, eps, alpha_prior, beta_prior)


if __name__ == "__main__":
    ap = argparse.ArgumentParser
    ap.add("--data")
    # ap.add("--alpha", default = ,type = float)
    # ap.add("--beta", default = ,type = float)

    args = ap.parse_args()
    main(args.data)