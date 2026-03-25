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
import torch
import torch.nn as nn
import argparse

# from data_loader import load_and_clean_data

def clean_occlusion_reason(df):
    #this is for when the visibility_category is 3 or 5 -> then there is no occlusion reason, so need to replace the blanks to prevent crashes

    df['occlusion_reason'] = df['occlusion_reason'].astype(str) #makes sure object is treated as string, handles NaNs

    # df.loc[~df['visibility_category'].isin([2.0, 4.0, 5.0]), 'occlusion_reason'] = "None"

    df.loc[df['visibility_category'] == 3.0, 'occlusion_reason'] = "off screen"
    df.loc[df['visibility_category'] == 1.0, 'occlusion_reason'] = 'visible'
    df.loc[df['visibility_category'] == 4.0, 'occlusion_reason'] = 'confused, too ambiguous'

    df.loc[df['occlusion_reason'].isna() | (df['occlusion_reason'] == 'nan'), 'occlusion_reason'] = "None" #just doing some double checking cleaning

    return df

def normalize_occlusion_reasons(df):
    #makes occlusion_reason order-invariant
    #Example: "self_occlusion, external_object" == "external_object, self_occlusion"

    df['occlusion_reason'] = df['occlusion_reason'].str.split(", ").apply(lambda x: ", ".join(sorted(set(item.strip() for item in x))) if isinstance(x, list) else "")
    #now we have a list of the objects, we want to sort, then concat back together

    return df

def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)
    df = clean_occlusion_reason(df)
    df = normalize_occlusion_reasons(df)

    return df

# def confidence_i (i: str, frame_n: str, df):
#     #input: joint i, frame_n (what frame we are at), df = pd.DataFrame
#     #output: confidence float for that joint i at frame_n

# def confidence_i_star (i: str, confidence_i: list, optical_flow_i: list, difference_i: float, std: float):

class LightweightTCN(nn.Module): #inherits from nn.Module (PyTorch's base class for neural networks)

    def __init__(self, num_joints = 17, hidden_channels = 64): #by default, we have 17 (COCO) joints, but the user can change this if they want.
        #by default, our internal representation uses 64 channels, but again, this is changeable
        super().__init__() #do the initialization that nn.Module needs to do, then do my custom initialization for LightweightTCN

        self.conv1 = nn.Conv1d( 
            # Take 17 joints over 16 frames. Look at each 5-frame window. Learn how to combine those joints into 64 new features. Output 64 features for all 16 frames.
            
            #takes in a 1D sequence, just a list of numbers, and slides a 1D filter over it, producing another 1D sequence
            num_joints, #input channels = num_joints = 17
            hidden_channels, #output channels = hidden_channels = 64 (internal representation)
            kernel_size = 5, #filter length, looks at 5 consecutive frames at a time
            dilation = 1, #stride of 1, normal, no skipping
            padding = 2 #adds padding on both sides so that output is the same size as the input
            #output: batch_size, 64 channels, 16 frames
        )

        self.conv2 = nn.Conv1d( 
            hidden_channels, #input: 64 channels, 16 frames
            hidden_channels, #output: 64 channels, 16 frames
            kernel_size = 5, #still looking at 5 frames at a time
            dilation = 2, #but now looking at every other frame
            padding = 4
        )

        self.tconv1 = nn.ConvTranspose1d(
            hidden_channels, #64 channels, 16 frames, etc.
            hidden_channels,
            kernel_size = 5, dilation = 1, padding = 2
        )

        self.tconv2 = nn.ConvTranspose1d(
            hidden_channels,
            num_joints,
            kernel_size = 5,
            dilation = 2,
            padding = 4
        )

        self.relu = nn.ReLU()

    def forward(self, x, mask = None):
        x = self.relu(self.conv1(x)) #max(0, x), non linear, so can now learn non-linear patterns, sparse, so lots of 0s, saves computation
        x = self.relu(self.conv2(x))
        x = self.relu(self.tconv1(x))
        x = torch.sigmoid(self.tconv2(x)) 

        if mask is not None: #if mask[i] = 1, keep x[i] (reliable joint, use smoothed output) #else, if mask[i] = 0, unreliable joint, remove, do not use.
            x = x * mask + (1 - mask) * x.detach() #so if mask[i]= 0, then 1 - 0 = 1, so x.detach(), which means don't compute gradients on this value, don't learn from it
        
        return x

def prepare_temporal_data (df, window_size = 16, stride = 8):
    #output should look like, per every 16 frames, per person:
    #[(confidence for joint 0, frame 1), (confidence for joint 1, frame 1), ..., (confidence for joint 16, frame 1)]
    #,...,
    #[(confidence for joint 0, frame 16), (confidence for joint 1, frame 16), ..., (confidence for joint 16, frame 16)]
    windows = []
    frame_ids = []
    
    # Group by person/track
    for track_id, group in df.groupby([' instance_id']):
        # Sort by frame within this track
        group = group.sort_values('frame_id').reset_index(drop=True)
        
        # Pivot: rows=frames, cols=joints
        pivot = group.pivot_table(
            index='frame_id',
            columns='joint_id',
            values=['mmpose_confidence', 'visibility_category', 'occlusion_severity'],
            aggfunc='first'
        )
        
        num_frames = len(pivot)
        
        # Sliding window over this person's sequence
        for start in range(0, num_frames - window_size + 1, stride):
            end = start + window_size
            window_data = pivot.iloc[start:end]
            windows.append(window_data)
            frame_ids.append(pivot.index[start:end].tolist())
    
    return windows, frame_ids



def main(csv_path, window_size=16, stride=8):
    """
    Main entry point for Phase 3: Temporal Smoothing.
    
    Args:
        csv_path: path to cleaned CSV
        window_size: temporal window size
        stride: sliding window stride
    """
    print("=" * 80)
    print("PHASE 3: TEMPORAL SMOOTHING (Cheng et al. inspired)")
    print("=" * 80)
    
    # Load and clean data
    print(f"\n1. Loading data from {csv_path}...")
    df = load_and_clean_data(csv_path)
    print(f"   ✓ Loaded {len(df)} rows")
    print(f"   ✓ Visibility categories: {df['visibility_category'].unique()}")
    print(f"   ✓ Unique occlusion reasons: {df['occlusion_reason'].nunique()}")
    
    # Inspect cleaned data
    print(f"\n2. Data quality check:")
    print(f"   - Total joints: {len(df)}")
    print(f"   - Visibility distribution:\n{df['visibility_category'].value_counts().sort_index()}")
    print(f"\n   - Occlusion reason distribution:\n{df['occlusion_reason'].value_counts()}")
    
    # Prepare temporal windows
    print(f"\n3. Preparing temporal windows (window_size={window_size}, stride={stride})...")
    windows, frame_ids = prepare_temporal_data(df, window_size=window_size, stride=stride)
    print(f"   ✓ Created {len(windows)} temporal windows")
    
    # Initialize model
    print(f"\n4. Initializing LightweightTCN...")
    model = LightweightTCN(num_joints=17, hidden_channels=64)
    print(f"   ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # (Phase 3.1) You can train the model here
    # (Phase 3.2) Or load pre-trained weights
    # (Phase 3.3) Or do inference on test set
    
    print("\n" + "=" * 80)
    print("Ready for temporal smoothing experiments!")
    print("=" * 80)
    
    return df, model, windows, frame_ids


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV with annotations")
    ap.add_argument("--window_size", type=int, default=16, help="Temporal window size")
    ap.add_argument("--stride", type = int, default = 8)
    args = ap.parse_args()
    df, model, windows, frame_ids = main(args.data, args.window_size, args.stride)

