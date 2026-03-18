import pandas as pd
import numpy as np
import argparse
from scipy.special import betaln   # log of the beta function B(α,β)


##negative log posterior J(alpha, beta)
def neg_log_posterior(alpha, beta, scores, eps: float = 1e-6, alpha_prior = 2.0, beta_prior = 2.0): 
    ##FOR OPTIMIZATION'S SAKE: consider placing a different distribution over alpha and beta parameters 
    # i.e., a gamma or hierarchical bayes distrbution
    #as opposed to the initial exponential distribution


    c = scores
    c = np.clip(c, eps, 1.0 - eps)
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


## MAP with weak Beta prior

## Logit Gaussian

## Jitter and clip

## Hierarchical / Empirical‑Bayes pooling

## Isotonic regression (non‑parametric)

## Simple Binning