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


ibatch_target = 64
fold_ratio = 16
ifolded_width = 32
iwidth = ifolded_width * fold_ratio
ibatch = math.ceil(ibatch_target / fold_ratio)
train_epochs = 8192
test_epochs = 256
pe_depth_multiplier = 1
pe_multiplier = 1.5
base_freq = 2.
device = torch.device("cuda")

print("Embedding Depth == ", ifolded_width)
print("Sequence Length == ", fold_ratio)

# x = torch.randn(ibatch, iwidth).to(device)

# gr = GroupedTransform(ifolded_width)
# x = gr(x)

# y = torch.arange(fold_ratio, dtype=torch.int32).expand(ibatch, fold_ratio).to(device)

# x = x.reshape(-1, ifolded_width)
# y = y.reshape(-1)
# assert(x.shape[0] == y.shape[0])

# print(x.shape)
# print(x)
# print(y.shape)
# print(y)

# indices = torch.randperm(int(y.shape[0]))
# print(indices)

# x = x[indices, :]
# y = y[indices]

# print(x)
# print(y)


gr = GroupedTransform(ifolded_width).to(device)
p = PositionalEncoding(ifolded_width*pe_depth_multiplier, fold_ratio + 1, pe_multiplier=pe_multiplier, base_freq=base_freq).to(device)
# p = BinaryPositionalEncoding(ifolded_width*pe_depth_multiplier, fold_ratio + 1, pe_multiplier=pe_multiplier).to(device)
# p = UniformRandomPositionalEncoding(ifolded_width*pe_depth_multiplier, fold_ratio + 1, pe_multiplier=pe_multiplier).to(device)


data = p.pe[0, :fold_ratio, :ifolded_width].cpu().numpy()
print(data.shape)

plt.title("PE Offsets")
plt.xlabel("Embedding")
plt.ylabel("Fold")
pe_plot = plt.imshow(data, cmap="RdBu", interpolation="nearest", aspect="auto")
plt.colorbar(pe_plot)
plt.savefig("encoding/pe_offsets.png", dpi=300)

# x = data[0, :]
# y = data[1:, :]

# print(x)
# print(y)

# exit(1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.af = AnyF1([ifolded_width], 20)
        self.fc1 = nn.Linear(ifolded_width, fold_ratio)
        # self.fc2 = nn.Linear(ifolded_width*2, fold_ratio)
        # self.fc3 = nn.Linear(ifolded_width, fold_ratio)
        # self.fc4 = nn.Linear(ifolded_width, fold_ratio)


    def forward(self, x):
        # x = self.af(x)
        x = F.relu(self.fc1(x))
        # x = F.relu(input + self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc2(x)
        return x


model = Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

y = torch.arange(fold_ratio, dtype=torch.int64).expand(ibatch, fold_ratio).to(device).reshape(-1)

if y.shape[0] > ibatch_target:
    y = y.reshape(-1, ibatch_target)
else:
    y = y.reshape(1, ibatch_target)

    # In this case batch size target is above the folding ratio
    # So each batch has multiple samples of each "category"
    # So reduce the training and test epochs to match the same ratio
    # train_epochs = math.ceil(train_epochs/(ibatch_target/fold_ratio))
    # test_epochs = math.ceil(test_epochs/(ibatch_target/fold_ratio))

nchunks = y.shape[0]
print("# of chunks == ", nchunks)
print("Batch size == ", y.shape[1])
print("Training epochs == ", train_epochs)
print("Test epochs == ", test_epochs)

model.train()
for epoch in trange(train_epochs):
    x = torch.randn(ibatch, iwidth).to(device)
    x = gr(x)
    if DO_PE_OFFSETS:
        x = p(x)
    x = x.reshape(-1, ibatch_target, ifolded_width)
    # print(y.shape)
    # print(x.shape)
    # exit(1)
    assert x.shape[0] == y.shape[0], "{} != {}".format(x.shape[0], y.shape[0])

    labels = y.clone()

    for ichunk in range(nchunks):
        # Shuffle x and y similarly
        indices = torch.randperm(int(y.shape[1]))
        data = x[ichunk, indices]
        target = labels[ichunk, indices]

        # print(data)
        # print(target)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # print('[{}/{}] -- Loss: {:.6f}\r'.format(epoch+1, train_epochs, loss.item(), end=''))
# print('\n')


model.eval()
correct = 0
for epoch in trange(test_epochs):
    x = torch.randn(ibatch, iwidth).to(device)
    x = gr(x)
    if DO_PE_OFFSETS:
        x = p(x)
    x = x.reshape(-1, ibatch_target, ifolded_width)
    assert x.shape[0] == y.shape[0], "{} != {}".format(x.shape[0], y.shape[0])

    labels = y.clone()

    for ichunk in range(nchunks):
        # Shuffle x and y similarly
        indices = torch.randperm(int(y.shape[1]))
        data = x[ichunk, indices]
        target = labels[ichunk, indices]

        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

print("Accuracy: {}/{} == {:.2f}".format(correct, y.shape[1]*test_epochs*nchunks, correct/(y.shape[1]*test_epochs*nchunks)*100.))
