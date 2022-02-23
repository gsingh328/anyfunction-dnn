import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from anyf import *

import numpy as np
import matplotlib.pyplot as plt


DO_PE_OFFSETS = True


torch.manual_seed(2022)
np.random.seed(2022)

iwidth = 32
iheight = 32
sequence_length = 2
vector_length = 32

train_epochs = 8192
test_epochs = 256


device = torch.device("cuda")


def shuffle_labels(labels):
    for i in range(labels.shape[0]):
        labels[i, :] = torch.randperm(int(labels.shape[1]))
    return labels


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.c1 = nn.Conv2d(vector_length, sequence_length, 3, stride=1, padding=1)


    def forward(self, x):
        x = self.c1(x)
        x = F.avg_pool2d(x, (iwidth, iheight))
        return torch.flatten(x, 1)


model = Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

p = PositionalEncoding(vector_length, sequence_length + 1, pe_multiplier=1.0, base_freq=10000.).to(device)
print(p.pe)
exit(1)

model.train()
for epoch in trange(train_epochs):
    x = torch.randn(sequence_length, vector_length, iwidth, iheight).to(device)
    labels = torch.randperm(sequence_length).to(device)
    pe_offset = p.pe[0, :sequence_length, :vector_length]
    offset = pe_offset[labels.reshape(-1), :].reshape(sequence_length, vector_length, 1, 1).to(device)

    if DO_PE_OFFSETS:
        x = x + offset 

    optimizer.zero_grad()

    output = model(x)

    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    # if (epoch+1) % 64 == 0:
    #     print('[{}/{}] -- Loss: {:.6f}\r'.format(epoch+1, train_epochs, loss.item(), end=''))


model.eval()
correct = 0
for epoch in trange(test_epochs):
    x = torch.randn(sequence_length, vector_length, iwidth, iheight).to(device)
    labels = torch.randperm(sequence_length).to(device)
    pe_offset = p.pe[0, :sequence_length, :vector_length]
    offset = pe_offset[labels.reshape(-1), :].reshape(sequence_length, vector_length, 1, 1).to(device)

    if DO_PE_OFFSETS:
        x = x + offset 

    output = model(x)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(labels.view_as(pred)).sum().item()

print("Accuracy: {}/{} == {:.2f}".format(correct, sequence_length*test_epochs, correct/(sequence_length*test_epochs)*100.))
