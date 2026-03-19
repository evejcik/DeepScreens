import pandas as pd
import numpy as np
import argparse
from sklearn.isotonic import IsotonicRegression

#For my documentation/comprehension:
#Isotonic refers to monotonic-increasing (or decreasing), but essentially means that we are fitting a non-decreasing function to data while
#minimizing our chosen loss function. We do not assume a particular parametric shape (i.e., linear, quadratic, Beta, ...) for the data.

#HOWEVER isotonic is very sensitive to outliers, it can overfit quite easily, thus works best for large datasets (usually 1000+)



##FUTURE RECOMMENDATION: For any future work where you add additional features (e.g., joint velocity, temporal context) and train an explicit classifier, 
# switch to CalibratedClassifierCV to obtain calibrated probabilities without reinventing the wheel.

def fit_global_isotonic(df):
    iso = IsotonicRegression(out_of_bounds = 'clip', increasing = True)
    iso.fit(df['mmpose_confidence'].values, df['label'].values)
    return iso


def main(df):
    df = pd.read_csv()


if __name__ == "__main__":
    ap = argparse.ArgParser()
    ap.add("--df")
    args = ap.parse_args()
    main(args.df)