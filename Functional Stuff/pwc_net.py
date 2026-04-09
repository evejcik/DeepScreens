# @InProceedings{Sun2018PWC-Net,
#   author    = {Deqing Sun and Xiaodong Yang and Ming-Yu Liu and Jan Kautz},
#   title     = {{PWC-Net}: {CNNs} for Optical Flow Using Pyramid, Warping, and Cost Volume},
#   booktitle = CVPR,
#   year      = {2018},
# }
# https://github.com/NVlabs/PWC-Net/tree/master/PyTorch


import torch
from pwcnet.pwcnet import PWCNet
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# Load the model (downloads pretrained weights automatically)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pwc_net = PWCNet().to(device)
pwc_net.eval()

def get_optical_flow_pwcnet(frame_t, frame_t1):
    """
    Compute optical flow using PWC-Net.
    
    Args:
        frame_t, frame_t1: numpy arrays, shape (H, W, 3), values in [0, 255], dtype uint8
    
    Returns:
        flow: numpy array, shape (H, W, 2)
    """
    # Convert to torch tensors and normalize to [-1, 1]
    img1 = torch.from_numpy(frame_t).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
    img2 = torch.from_numpy(frame_t1).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
    
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    with torch.no_grad():
        flow = pwc_net(img1, img2)  # shape: (1, 2, H, W)
    
    # Convert back to numpy
    flow = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
    
    return flow