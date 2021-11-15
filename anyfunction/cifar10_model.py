from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from anyf import AnyF1


class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
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


class BasicCNN_AnyF(nn.Module):
    def __init__(self):
        super(BasicCNN_AnyF, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        # self.conv2_af = AnyF1((32, 30, 30), 20)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.fc1_af = AnyF1([12544], 20)
        self.fc1 = nn.Linear(12544, 10)
        # self.fc2_af = AnyF1([128], 20)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        # x = self.conv2_af(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)

        # x = self.fc1_af(x)

        x = self.fc1(x)
        # x = F.relu(x)

        # x = self.fc2_af(x)

        # x = self.fc2(x)
        return x


class BasicMLP(nn.Module):
    def __init__(self):
        super(BasicMLP, self).__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BasicMLP_AnyF(nn.Module):
    def __init__(self):
        super(BasicMLP_AnyF, self).__init__()
        self.af1 = AnyF1([128], 20, actv_out=True)
        self.fc1 = nn.Linear(128, 512)
        # self.af2 = AnyF1([128], 20, actv_out=True)
        # self.fc2 = nn.Linear(128, 128)
        # self.af3 = AnyF1([128], 20, actv_out=True)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.af1(x)
        x = F.relu(self.fc1(x))

        # x = self.af2(x)
        # x = F.relu(self.fc2(x))

        # x = self.af3(x)
        x = self.fc3(x)
        return x


def Net(*args, **kwargs):
    # net = BasicCNN(*args, **kwargs)
    net = BasicCNN_AnyF(*args, **kwargs)
    # net = BasicMLP(*args, **kwargs)
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total parameters: {}".format(pytorch_total_params))
    return net
