#2D Convolutional Temporal Network
#input: masked position vector. a vector of length 2K (how many joints there are, and their x and y positions) with values 1 or 0, to indicate whether or not that joint is reliable

# The loss of the 2D TCN is formulated as:  Loss = 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


from confidence_scores import min_max_scalar, position_vec, binary_map


def loss(X_tilde_masked, C_b, X_tilde, net):
    #input: X_tilde_masked = output from binary_map, C^b * X_tilde, size (2K, )
    #C_b = 1/0 binary map, also output from binary_map, size (2K, )
    #output: loss value, to be minimized
    X = X_tilde

    # Convert to torch tensors and add batch dimension
    X_masked_t = torch.from_numpy(X_tilde_masked).float().unsqueeze(0)   # (1, 2K)
    C_b_t      = torch.from_numpy(C_b).float().unsqueeze(0)       # (1, 2K)
    X_gt_t     = torch.from_numpy(X_tilde).float().unsqueeze(0)      # (1, 2K)

    X_hat = net(X_masked_t)    # (1, 2K)
    diff = (X_hat * X) - C_b_t
    loss_val = torch.mean(diff.pow(2))
    return loss_val


class Net(nn.Module): #1D CNN for temporal consistency
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2D(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.(16 * 5 * 5,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)


net = Net()

def main(data_path, masked_joints, net):
    #X_tilde = [x_1, y_1, ..., x_K, y_K]
    #X_masked_tilde = X_tilde * C_b
    #
    csv = pd.read_csv(Path(data_path))
    
    X_tilde = position_vec(csv)
    C_b, X_hat = binary_map(csv['mmpose_co'], c_i, threshold)
    loss(X_masked_tilde, df, X_tilde, net)


if __name__ == main():
    ap.argparse.ArgumentParser()    
    ap.add_argument('df')
    ap.add_argument("data_path")
    ap.add_argument("threshold", type = int, default = 0.3) #as in cited paper

    args = ap.parse_args()
    main(df)