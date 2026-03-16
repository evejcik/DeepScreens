import cv2
import os
from pathlib import Path
import json
import numpy as np
import argparse
import pandas as pd

# create vector of x,y confidence vectors, generated from tabular data from hand-annotated data
# pass this vector to optical flow calculation

# Add a variance head to your 2‑D model (predict a per‑joint log‑σ²) and train it with a negative‑log‑likelihood loss. 
# The variance then serves the same purpose as the heat‑map’s spread.

def min_max_scaler(confidence_vec): 
    #input: vector of confidence scores
    #output: normalized vector, between 0 and 1

    v_min = confidence_vec.min()
    v_max = confidence_vec.max()
    denom = v_max - v_min
    if denom == 0:
        return pd.Series(0.0, index=series.index)
    
    return (confidence_vec - v_min) / denom
    

def position_vec(df):
    #input: tabular dataframe
    #output: normalized x,y pairs in a vector of confidence scores

    x = df['x'].astype(float)
    y = df['y'].astype(float)

    X = min_max_scalar(pd.Series(x))
    Y = min_max_scalar(pd.Series(y))

    c_i = list(zip(X, Y))

    return c_i

def binary_map(confidence_vec, threshold):
    #input: confidence vector, position vector, threshold
    #output: a third vector, with whether or not the confidence is higher than threshold -> which joints are reliable
    reliability_map = list(map(lambda x: x >= threshold))
    return reliability_map

def main(data_path, threshold):
    csv = pd.read_csv(Path(data_path))
    
    c_i = position_vec(csv)
    binary_map(csv['mmpose_co'], c_i, threshold)




if __name__ == "main":
    ap.argparse.ArgumentParser()
    ap.add_argument("data_path")
    ap.add_argument("threshold", type = int, default = 0.3)

    args = ap.parse_args()
    main(data_path, threshold)