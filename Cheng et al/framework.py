# By employing estimated 2D confidence heatmaps of keypoints and an optical-flow consistency constraint, we filter out the
# unreliable estimations of occluded keypoints. When occlusion occurs, we have incomplete 2D keypoints and feed
# them to our 2D and 3D temporal convolutional networks (2D and 3D TCNs) that enforce temporal smoothness to
# produce a complete 3D pose. By using incomplete 2D keypoints, instead of complete but incorrect ones, our networks
# are less affected by the error-prone estimations of occluded keypoints.

# General framework:semi-supervised method
# Human Detection & Keypoints Estimation (done)
# Temporal Convolution for 2D pose
# Temporal Convolution for 3D pose

# 3D pose keypoints are then re-projected to 2D -> these 2D keypoints are used as a form of regularization to the 2D keypoints MSE loss.

# Network 1: outputs the estimated 2D locations of the keypoints for each bounding box of a person -> mmpose already does this for us. 
# Network 2: These maps are combined with optical flow to assess whether the predicted keypoints are occluded. 
# Filter out occluded keypoints and then feed the potentially incomplete 2D keypoints to our second and third networks which are both temporal convolutional networks
# (2D and 3D TCNs respecively) to enforce temporal smoothness.

# The 3D TCN takes potentially incomplete 2D keypoints as input, and requires pairs of a 3D pose and a 2D pose with occlusion labels during training. 
# We also add in a pose regularization term to penalize violations of keypoints that are estimated as unoccluded, contradicting the ground-truth labels.

# WE WANT TO PRODUCE LOW C_I FOR OCCLUDED KEYPOINTS. (confidence for i ∈ [1, K] and K is the number of predefined keypoints)
# We further apply optical flow to p_hat_i (estimated position of joint i)
# The flow vector is o_i

# Additionally, we need to process the location difference of keypoint i in the neighboring frames defined as d_i. 

# Final confidence score for p_i = C*_i = C_i * exp(- ||o_i - d_i||^2_2)/(2 * std ^2)
# If C*_i is smaller than a tuneable threshold p_i is labeled as an occluded keypoint.

import pandas as pd
import numpy as np

from data_loader import load_and_clean_data

def confidence_i (i: str, frame_n: str, df):
    #input: joint i, frame_n (what frame we are at), df = pd.DataFrame
    #output: confidence float for that joint i at frame_n

def confidence_i_star (i: str, confidence_i: list, optical_flow_i: list, difference_i: float, std: float):

class LightweightTCN(nn.Module):

    def __init__(self, num_joints = 17, hidden_channels = 64): #by default, we have 17 (COCO) joints, but the user can change this if they want.
        #by default, our internal representation uses 64 channels, but again, this is changeable
        super().__init__()

    self.conv1 = nn.Conv1d( #takes in a 1D sequence, just a list of numbers, and slides a 1D filter over it, producing ianother 1D sequence
        num_joints, hidden_channels,
        kernel_size = 5, dilation = 1, padding = 2
    )

    self.conv2 = nn.Conv1d( #
        hidden_channels, hidden_channels,
        kernel_size = 5, dilation = 2, padding = 4
    )

    self.tconv1 = nn.ConvTranspose1d(
        hidden_channels, hidden_channels,
        kernel_size = 5, dilation = 1, padding = 2
    )

    self.tconv2 = nn.Conv
