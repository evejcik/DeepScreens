#for each joint, randomly flip 10-20% of visibility labels
#compute soft labels = soft_label[i] = sum(annotator_votes[i]) / num_annotators
#for ambiguous joints (extremely low confidence, visibility = 4) -> treat as explicitly uncertain labels


import pandas as pd
import numpy as np
import argparse
import random
import math

#read in data, already annotated
def flip_visibility_category(data, pct = 20):
    df = pd.read_csv(data)

    n_change = round(pct / 100 * df.shape[0]) #how many rows to change
    floats = np.random.uniform(1.0, 5.0, size = n_change) #just having fun trying to make everything random lol
    binary = random.randint(0,1)
    rows_to_flip = np.random.randint(0,df.shape[0], size = n_change) #creates array of random values, 20% of len long

    new_vals = np.where(binary, np.floor(floats), np.ceil(floats)).astype(int)

    df[rows_to_flip, 'annotator_votes'] = floats.astype(int)
    return df['annotator_votes']

def soft_label(df, n_change):
    sum_agreement = np.sum(df['annotator_votes'] == df['visibility_category'])
    annotator_vals = {df.groupby(['annotator_votes']).sum()}
    df['soft_label'] = annotator_vals[val] / n_change
    

