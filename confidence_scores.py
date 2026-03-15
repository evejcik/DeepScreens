import cv2
import os
from pathlib import Path
import json
import numpy as np
import argparse
import pandas as pd

def confidence_vec(data_path):
    confidenceX = data_path.loc['x']
    confidenceY = data_path.loc['y']

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