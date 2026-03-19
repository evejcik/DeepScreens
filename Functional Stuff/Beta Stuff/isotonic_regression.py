import pandas as pd
import numpy as np
import argparse
from sklearn.isotonic import IsotonicRegression

#For my documentation/comprehension:
#Isotonic refers to monotonic-increasing (or decreasing), but essentially means that we are fitting a non-decreasing function to data while
#minimizing our chosen loss function. We do not assume a particular parametric shape (i.e., linear, quadratic, Beta, ...) for the data.






def main(df):
    df = pd.read_csv()


if __name__ == "__main__":
    ap = argparse.ArgParser()
    ap.add("--df")
    args = ap.parse_args()
    main(args.df)