#2D Convolutional Temporal Network
#input: masked position vector. a vector of length 2K (how many joints there are, and their x and y positions) with values 1 or 0, to indicate whether or not that joint is reliable

# The loss of the 2D TCN is formulated as:  Loss = 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from confidence_scores import X_hat, C_b


def loss(X_hat, C_b, df):
    #input: masked joint vector (length 2K)
    #output: loss value, to be minimized
    X = df['x']
    masked_coords = X_hat.reshape(-1)
    term_1 = Net.forward(masked_coords)    
    loss = C_b * (term_1 - X)
    loss_squared = math.pow(loss, 2)
    return loss_squaredx


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