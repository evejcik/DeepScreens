import pandas as pd
import numpy as np
import argparse


##negative log posterior J(alpha, beta)
def neg_log_posterior(alpha, beta, scores):
    c = scores
    c = np.clip(c, float(1e-6), 1.0 - (1e-6) )
    N = len(scores)


## MAP with weak Beta prior

## Logit Gaussian

## Jitter and clip

## Hierarchical / Empirical‑Bayes pooling

## Isotonic regression (non‑parametric)

## Simple Binning