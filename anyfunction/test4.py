import torch
import torch.nn as nn
from anyf import *

b = 2
n = 49
g = 4
h = 5

x = (torch.randn(b, n) * 10).floor_()
l = GroupedTransform(g)
y = l(x)

print(x.shape)
# print(x)

# print(y.shape)
# print(y)

p = PositionalEncoding(g, 100)
z = p(y)

print(z.shape)
print(z)

# w = (torch.randn(1, g, 6) * 10).floor_()

# print(w.shape)
# print(w)

# a = torch.matmul(z, w)

# fc1 = nn.Linear(g, h)
# a = fc1(z)

af = AnyF2([1, 1, 4], 20)
a = af(z)

print(a.shape)
print(a)

