import math
import torch
import torch.nn.functional as F
from torch.nn import init


# i = 4
# h = 5

# x = (torch.rand(2, i, 1) * 10).round()
# print("x:" + "-" * 40)
# print(x.shape)
# print(x)

# w = (torch.rand(1, i, h) * 10).round()
# print("w:" + "-" * 40)
# print(w.shape)
# print(w)

# b = (torch.rand(1, i, h) * 10).round()
# print("b: " + "-" * 40)
# print(b.shape)
# print(b)

# y = x.mul(w) + b
# print("y:" + "-" * 40)
# print(y.shape)
# print(y)

# w2 = (torch.rand(1, i, h) * 10).round()
# print("w2:" + "-" * 40)
# print(w2.shape)
# print(w2)

# b2 = (torch.rand(1, i) * 10).round()
# print("b2: " + "-" * 40)
# print(b2.shape)
# print(b2)

# y2 = y.mul(w2).sum(dim=-1) + b2
# print("y2:" + "-" * 40)
# print(y2.shape)
# print(y2)

# print("\n" + "="*40)

# input_size=1200
# hidden_size=8000

# w = torch.randn(hidden_size,input_size) * math.sqrt(3/input_size)
# x = torch.randn(128, input_size)
# y = F.relu(F.linear(x, w))
# print(x.var(), y.var())

# print("\n" + "="*40)

b = 128
i = 12000
h = 10

x = torch.empty(b, i, 1)
init.normal_(x)
w = torch.empty(1, i, h)
# init.normal_(w, std=math.sqrt(2.5))
init.normal_(w, std=math.sqrt(1/h))
# y = F.relu(x.mul(w))
y = F.relu(x.mul(w).sum(dim=-1))
y = x.mul(w).sum(dim=-1)

print(x.var(), y.var())
