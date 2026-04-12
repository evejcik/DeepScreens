#from Gonçalves — Savitzky-Golay filter for trust joints.


import pandas as pd
import numpy as np
import argparse
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from kalman import KalmanFilter, apply_kalman_to_group, apply_kalman_filter


#need to get long data csv
#need to get json

# def data_loader(csv):
#     df = pd.read(csv)


def sav_golay(df, k, p):
    #k = size of rolling window
    #p = degree of polynomial used to fit points
    #need tominimize the sum of the squares of the differences between the actual data points y_i and the polynomial values y_ih

    #we need to get the coordinates for (x,y) for each film, instance_id, frame_id, joint_id
    # def savgol_filter_mine(df, k, p):

    #only apply to trusted points -> to smooth between joints for aesthetics essentially.
    x = df[df['reliability_category_int'] == 0]['x']
    y = df[df['reliability_category_int'] == 0]['y']
    df['x_smooth'] = savgol_filter(x, window_length = k, polyorder = p)
    df['y_smooth'] = savgol_filter(y, window_length = k, polyorder = p)
    return df


def main(csv):
    df = pd.read(csv)
    df = sav_golay(df, k = 7, p = 10) #idk, guessing at these, finetune later
    df = apply_kalman_filter(df)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")

    args = ap.parse_args()
    main(
        args.csv
    )

