#2D Convolutional Temporal Network
#input: masked position vector. a vector of length 2K (how many joints there are, and their x and y positions) with values 1 or 0, to indicate whether or not that joint is reliable

# The loss of the 2D TCN is formulated as:  Loss = 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from confidence_scores import X_tilde_masked, C_b


def loss(X_tilde_masked, C_b, df):
    #input: X_tilde_masked = output from binary_map, C^b * X_tilde, size (2K, )
    #C_b = 1/0 binary map, also output from binary_map, size (2K, )
    #output: loss value, to be minimized
    X = df['x']

    # Convert to torch tensors and add batch dimension
    X_masked_t = torch.from_numpy(X_tilde_masked).float().unsqueeze(0)   # (1, 2K)
    C_b_t      = torch.from_numpy(C_b).float().unsqueeze(0)       # (1, 2K)
    X_gt_t     = torch.from_numpy(X_gt).float().unsqueeze(0)      # (1, 2K)

    X_hat = net(X_masked_t)    # (1, 2K)
    diff = (X_hat * C_b) - X
    loss_val = torch.mean(diff.pow(2))
    return loss_val


class Net(nn.Module):
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

def main(df, masked_joints):
    loss(masked_joints)


if __name__ == main():
    ap.argparse.ArgumentParser()    
    ap.add_argument('df')

    args = ap.parse_args()
    main(df)