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



def confidence_vec(data_path):
    confidenceX = data_path.loc['x']
    confidenceY = data_path.loc['y']

    c_i = list(zip(confidenceX, confidenceY))

    for x,y in c_i:
        scaled_x = (x - c_i[x].min()) / (c_i[x].max() - c_i[x].min())
        scaled_y = (y - c_i[y].min()) / (c_i[y].max() - c_i[y].min())
    return c_i



def main(data_path):
    csv = pd.read_csv(Path(data_path))

   c_i = confidence_vec(csv)


if __name__ == "main":
    ap.argparse.ArgumentParser()
    ap.add_argument("data_path")

    args = ap.parse_args()
    main(data_path)