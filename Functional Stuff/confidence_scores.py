# Y. Cheng, B. Yang, B. Wang, Y. Wending and R. Tan, "Occlusion-Aware Networks for 3D Human Pose Estimation in Video," 2019 IEEE/CVF International Conference on Computer Vision (ICCV), Seoul, Korea (South), 2019, pp. 723-732, doi: 10.1109/ICCV.2019.00081.
# keywords: {Three-dimensional displays;Two dimensional displays;Pose estimation;Heating systems;Feeds;Training},

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

def min_max_scalar(confidence_vec): 
    #input: vector of confidence scores
    #output: normalized vector, between 0 and 1

    # Convert to a NumPy array of floats (works for both Series and ndarray)
    conf = np.asarray(confidence_vec, dtype = np.float32)

    v_min = conf.min()
    v_max = conf.max()
    denom = v_max - v_min
    if denom == 0: #avoid dividing by 0 error
        return np.zeros_like(conf, dtype=np.float32)
    
    return (conf - v_min) / denom
    

def position_vec(df):
    #input: tabular dataframe
    #output: normalized x,y pairs in a vector of confidence scores

    x = df['x'].astype(float)
    y = df['y'].astype(float)

    x_norm = min_max_scalar(x)
    y_norm = min_max_scalar(y)

    X = np.empty(2 * len(x_norm), dtype=np.float32) #normalized coordinates??
    X[0::2] = x_norm #even spots are x
    X[1::2] = y_norm #odd spots are y

    return X

def binary_map(confidence_vec, X_tilde, threshold):
    #input: confidence vector, size K, threshold
    #confidence_vec = confidences, size 17
    #X_tilde = ground truth, size 17 x 2 (x,y per joint)
    #output: C_b, another vector, size 2K, with whether or not the confidence is higher than threshold -> which joints are reliable, indicated in 1s and 0s
    reliability_map = [float(x >= threshold) for x in confidence_vec]
    mask = np.asarray(reliability_map, dtype=float)   # shape (K,)
    C_b = np.repeat(mask, 2)           # shape (2*K,) #C_b

    X_tilde_masked = C_b.T * X_tilde #the coordinates have been masked now
    return C_b, X_tilde_masked

    
def main(data_path, threshold):
    csv = pd.read_csv(Path(data_path))
    
    c_i = position_vec(csv)
    C_b, X_hat = binary_map(csv['mmpose_co'], c_i, threshold)
    

if __name__ == "main":
    ap.argparse.ArgumentParser()
    ap.add_argument("data_path")
    ap.add_argument("threshold", type = int, default = 0.3) #as in cited paper

    args = ap.parse_args()
    main(data_path, threshold)