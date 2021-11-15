from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from anyf import AnyF1, GroupedTransform, PositionalEncoding, AnyF2


class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class BasicMLP(nn.Module):
    def __init__(self):
        super(BasicMLP, self).__init__()
        self.fc1 = nn.Linear(1*14*14, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BasicMLP_AnyF(nn.Module):
    def __init__(self):
        super(BasicMLP_AnyF, self).__init__()
        # self.fc1 = nn.Linear(1*7*7, 32)
        # self.fc2 = nn.Linear(32, 10)
        # self.af = AnyF1([1*7*7], 20)

        self.gr = GroupedTransform(4)
        self.p = PositionalEncoding(4, 1000)

        self.af1 = AnyF2([1, 4], 100)
        self.af2 = AnyF2([1, 64], 20)

        self.fc1 = nn.Linear(4, 64)
        # self.fc2 = nn.Linear(64, 512)
        self.nrm = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 10)


    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        # x = torch.flatten(x, 1)

        # x = self.af(x)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)

        x = self.gr(x)
        x = self.p(x)
        x = self.af1(x)

        x = F.gelu(self.fc1(x))
        # x = self.af2(x)
        # x = F.gelu(self.fc2(x))
        x = x.sum(dim=0)
        x = self.nrm(x)
        x = self.af2(x)

        x = self.fc3(x)

        return x

def Net(*args, **kwargs):
    # net = BasicCNN(*args, **kwargs)
    # net = BasicMLP(*args, **kwargs)
    net = BasicMLP_AnyF(*args, **kwargs)
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total parameters: {}".format(pytorch_total_params))
    return net

