import numpy as np
import pandas as pd
import argparse


##to be used when a joint has very few visible samples (i.e., Ramona right ankle only has 9 visible samples, I'm sure some down the road will have 0 visible samples for ankles, feet, etc.)''

#first need to estimate the global prior - all confidence scores from csv, for all visibility classes

#apply jitter and compute mom start values for a and b 
#optionally refine those a_0 and b_0 values with an MAP optimisation using the same weak prior (2,2) as in beta MAP.py. the result a_0, b_0 is the global beta.
#because we are using ALL data, the global variance is never zero, so the method of moments (mom) estimate is stable.

#once we have the global prior, we can pass the global parameters of this distribution into the loss function for a given point, especially if the number of samples is low for that specific one.

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--df")

    args = ap.parse_args()
    main(args.df)

