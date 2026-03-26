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

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from ..Data_Exploration.data_loader import load_and_clean_data

# def clean_occlusion_reason(df):
#     #this is for when the visibility_category is 3 or 5 -> then there is no occlusion reason, so need to replace the blanks to prevent crashes

#     df['occlusion_reason'] = df['occlusion_reason'].astype(str) #makes sure object is treated as string, handles NaNs

#     # df.loc[~df['visibility_category'].isin([2.0, 4.0, 5.0]), 'occlusion_reason'] = "None"

#     df.loc[df['visibility_category'] == 3.0, 'occlusion_reason'] = "off screen"
#     df.loc[df['visibility_category'] == 1.0, 'occlusion_reason'] = 'visible'
#     df.loc[df['visibility_category'] == 4.0, 'occlusion_reason'] = 'confused, too ambiguous'

#     df.loc[df['occlusion_reason'].isna() | (df['occlusion_reason'] == 'nan'), 'occlusion_reason'] = "None" #just doing some double checking cleaning

#     return df

# def normalize_occlusion_reasons(df):
#     #makes occlusion_reason order-invariant
#     #Example: "self_occlusion, external_object" == "external_object, self_occlusion"

#     df['occlusion_reason'] = df['occlusion_reason'].str.split(", ").apply(lambda x: ", ".join(sorted(set(item.strip() for item in x))) if isinstance(x, list) else "")
#     #now we have a list of the objects, we want to sort, then concat back together

#     return df

# def load_and_clean_data(csv_path):
#     df = pd.read_csv(csv_path)
#     df = clean_occlusion_reason(df)
#     df = normalize_occlusion_reasons(df)

#     return df

class LightweightTCN(nn.Module): #inherits from nn.Module (PyTorch's base class for neural networks)

    def __init__(self, num_joints = 7, hidden_channels = 64): #by default, we have 17 (COCO) joints, but the user can change this if they want.
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



def convert_windows_to_tensors(windows, feature_input='mmpose_confidence', 
                               feature_target='visibility_category'):
    """
    Convert pandas window DataFrames to PyTorch tensors.
    
    Args:
        windows: list of pivot table windows (from prepare_temporal_data)
        feature_input: which column to use as input (e.g., 'mmpose_confidence')
        feature_target: which column to use as target (e.g., 'visibility_category')
    
    Returns:
        X: [num_windows, 17 joints, 16 frames] input tensors
        y: [num_windows, 17 joints, 16 frames] target tensors
    """
    X = []
    y = []
    
    for window in windows:
        # Extract input feature (mmpose_confidence)
        # window structure: multi-level columns (feature_name, joint_id)
        input_data = window[feature_input].values  # [16 frames, 17 joints]
        input_data = input_data.T  # [17 joints, 16 frames]
        input_data = torch.FloatTensor(input_data)
        X.append(input_data)
        
        # Extract target feature (visibility_category)
        target_data = window[feature_target].values  # [16 frames, 17 joints]
        # Convert visibility_category (1-5) to probability (0-1)
        # 1 = visible → 1.0
        # 2 = occluded → 0.0
        # 3 = off-screen → 0.0
        # 4 = ambiguous → 0.5
        # 5 = hallucinated → 0.0
        visibility_to_prob = {
            1.0: 1.0,   # visible
            2.0: 0.0,   # occluded
            3.0: 0.0,   # off-screen
            4.0: 0.5,   # ambiguous (uncertain)
            5.0: 0.0    # hallucinated
        }
        target_data = torch.FloatTensor(
            [[visibility_to_prob.get(val, 0.0) for val in row] for row in target_data]
        )
        target_data = target_data.T  # [17 joints, 16 frames]
        y.append(target_data)
    
    # Stack all windows
    X = torch.stack(X)  # [num_windows, 17 joints, 16 frames]
    y = torch.stack(y)  # [num_windows, 17 joints, 16 frames]
    
    return X, y

class VisibilityDataset(Dataset):
    """
    PyTorch Dataset for temporal visibility smoothing.
    """
    
    def __init__(self, X, y):
        """
        Args:
            X: [num_windows, 17 joints, 16 frames] input tensors
            y: [num_windows, 17 joints, 16 frames] target tensors
        """
        self.X = X
        self.y = y
    
    def __len__(self):
        """Return number of samples."""
        return len(self.X)
    
    def __getitem__(self, idx):
        """Return one sample."""
        return self.X[idx], self.y[idx]

def train_tcn(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-3):
    """
    Train the LightweightTCN.
    
    Args:
        model: LightweightTCN instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: number of training epochs
        learning_rate: learning rate for optimizer
    
    Returns:
        train_losses: list of training losses per epoch
        val_losses: list of validation losses per epoch
    """
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss function: Mean Squared Error
    criterion = torch.nn.MSELoss()
    
    # Optimizer: Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    print("\n" + "=" * 80)
    print("TRAINING LIGHTWEIGHTTCN")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Num epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 80 + "\n")
    
    for epoch in range(num_epochs):
        # ===== Training Phase =====
        model.train()  # Set model to training mode
        train_loss = 0.0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            
            # Compute loss
            loss = criterion(y_pred, y_batch)
            
            # Backward pass
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
            
            train_loss += loss.item()
        
        # Average training loss for this epoch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # ===== Validation Phase =====
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        
        with torch.no_grad():  # Don't compute gradients during validation
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Forward pass
                y_pred = model(X_batch)
                
                # Compute loss
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        # Average validation loss for this epoch
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80 + "\n")
    
    return train_losses, val_losses

def main(csv_path, window_size=16, stride=8, verbose=True, 
         num_epochs=20, batch_size=16, val_split=0.2):
    """
    Main entry point for Phase 3: Temporal Smoothing.
    
    Args:
        csv_path: path to CSV with annotations
        window_size: temporal window size (frames)
        stride: sliding window stride
        verbose: print diagnostics
        num_epochs: number of training epochs
        batch_size: batch size for training
        val_split: fraction of data to use for validation (0.2 = 20%)
    """
    print("=" * 80)
    print("PHASE 3: TEMPORAL SMOOTHING (Cheng et al. inspired)")
    print("=" * 80)
    
    # ===== Step 1: Load and Clean Data =====
    print(f"\n1. Loading data from {csv_path}...")
    df = load_and_clean_data(csv_path)
    print(f"   ✓ Loaded {len(df)} rows")
    
    # if verbose:
    #     print_diagnostics(df)
    
    # ===== Step 2: Prepare Temporal Windows =====
    print(f"\n2. Preparing temporal windows (window_size={window_size}, stride={stride})...")
    windows, frame_ids = prepare_temporal_data(df, window_size=window_size, stride=stride)
    print(f"   ✓ Created {len(windows)} temporal windows")
    
    # ===== Step 3: Convert to PyTorch Tensors =====
    print(f"\n3. Converting windows to PyTorch tensors...")
    X, y = convert_windows_to_tensors(
        windows, 
        feature_input='mmpose_confidence',
        feature_target='visibility_category'
    )
    print(f"   ✓ X shape: {X.shape}")
    print(f"   ✓ y shape: {y.shape}")
    print(f"   ✓ Input range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"   ✓ Target range: [{y.min():.3f}, {y.max():.3f}]")
    
    # ===== Step 4: Train/Validation Split =====
    print(f"\n4. Splitting data (val_split={val_split})...")
    num_samples = len(X)
    num_val = int(num_samples * val_split)
    num_train = num_samples - num_val
    
    # Random split
    indices = torch.randperm(num_samples).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    
    print(f"   ✓ Train set: {len(X_train)} samples")
    print(f"   ✓ Val set: {len(X_val)} samples")
    
    # ===== Step 5: Create DataLoaders =====
    print(f"\n5. Creating DataLoaders (batch_size={batch_size})...")
    train_dataset = VisibilityDataset(X_train, y_train)
    val_dataset = VisibilityDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"   ✓ Train batches: {len(train_loader)}")
    print(f"   ✓ Val batches: {len(val_loader)}")
    
    # ===== Step 6: Initialize Model =====
    print(f"\n6. Initializing LightweightTCN...")
    model = LightweightTCN(num_joints=7, hidden_channels=64)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model parameters: {num_params:,}")
    
    # ===== Step 7: Train Model =====
    print(f"\n7. Training model...")
    train_losses, val_losses = train_tcn(
        model, 
        train_loader, 
        val_loader,
        num_epochs=num_epochs,
        learning_rate=1e-3
    )
    
    # ===== Step 8: Results =====
    print(f"\n8. Training Results:")
    print(f"   ✓ Final train loss: {train_losses[-1]:.6f}")
    print(f"   ✓ Final val loss: {val_losses[-1]:.6f}")
    print(f"   ✓ Best val loss: {min(val_losses):.6f} (epoch {val_losses.index(min(val_losses)) + 1})")
    
    print("\n" + "=" * 80)
    print("Ready for evaluation!")
    print("=" * 80)
    
    return model, train_losses, val_losses, X_val, y_val, val_loader


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV with annotations")
    ap.add_argument("--window_size", type=int, default=16, help="Temporal window size")
    ap.add_argument("--stride", type=int, default=8, help="Sliding window stride")
    ap.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size")
    ap.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    args = ap.parse_args()
    
    model, train_losses, val_losses, X_val, y_val, val_loader = main(
        csv_path=args.data,
        window_size=args.window_size,
        stride=args.stride,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        val_split=args.val_split
    )

