import torch
import torch.nn as nn
import torch.nn.functional as F
from anyf import AnyF1


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
        # self.af1 = AnyF1(49, 20)
        self.fc1 = nn.Linear(7*7*1, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        # x = self.af1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def Net(*args, **kwargs):
    # net = BasicCNN(*args, **kwargs)
    net = BasicMLP(*args, **kwargs)
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total parameters: {}".format(pytorch_total_params))
    return net

