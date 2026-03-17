import cv2
import numpy as np

import argparse
import pandas as pd
import pathlib as Path

def get_optical_flow(frame_t, frame_t1):
    """
    Compute optical flow between two consecutive frames.
    
    Args:
        frame_t:  current frame (RGB, uint8, shape: HxWx3)
        frame_t1: next frame (RGB, uint8, shape: HxWx3)
    
    Returns:
        flow: optical flow field (shape: HxWx2, where flow[:,:,0] is x-displacement, 
              flow[:,:,1] is y-displacement)
    """
    # Convert to grayscale
    gray_t = cv2.cvtColor(frame_t, cv2.COLOR_RGB2GRAY)
    gray_t1 = cv2.cvtColor(frame_t1, cv2.COLOR_RGB2GRAY)
    
    # Compute dense optical flow (Farneback algorithm)
    flow = cv2.calcOpticalFlowFarneback(
        gray_t, gray_t1,
        flow=None,
        pyr_scale=0.5,      # image scale between pyramid levels
        levels=3,           # number of pyramid levels
        winsize=15,         # averaging window size
        iterations=3,       # iterations per level
        poly_n=5,           # size of pixel neighborhood
        poly_sigma=1.2,     # standard deviation for derivatives
        flags=0
    )
    
    return flow  # shape: (H, W, 2)