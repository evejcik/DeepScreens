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

def min_max_scaler(vec):
    v_min = vec.min()
    v_max = vec.max()
    denom = v_max - v_min
    # if denom == 0:
    
    return vec - v_min / denom
    

def confidence_vec(df):
    x = df['x'].astype(float)
    y = df['y'].astype(float)

    confidenceX = min_max_scalar(x)
    confidenceY = min_max_scalar(y)

    c_i = list(zip(confidenceX, confidenceY))

    return c_i



def main(data_path):
    csv = pd.read_csv(Path(data_path))

   c_i = confidence_vec(csv)


if __name__ == "main":
    ap.argparse.ArgumentParser()
    ap.add_argument("data_path")

    args = ap.parse_args()
    main(data_path)