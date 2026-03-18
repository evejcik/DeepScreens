import pandas as pd
import numpy as np
import argparse
from scipy.special import betaln   # log of the beta function B(α,β)


##negative log posterior J(alpha, beta)
def neg_log_posterior(alpha, beta, scores, eps: float = 1e-6, alpha_prior, beta_prior):
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

    sum_term = alpha_1 * sigma_log_ci + beta_1 * sigma_log_1_ci - n_log_beta
    final = sum_term + (alpha - alpha_prior) + (beta - beta_prior)


## MAP with weak Beta prior

## Logit Gaussian

## Jitter and clip

## Hierarchical / Empirical‑Bayes pooling

## Isotonic regression (non‑parametric)

## Simple Binning