import torch
from anyf import AnyF1


# b = 2
# i = 4
# h = 5


# af1 = AnyF1(i, h)

# with torch.no_grad():
#     af1.weight1.mul_(10.).round_()
#     af1.weight2.mul_(10.).round_()
#     af1.bias1.mul_(10.).round_()
#     af1.bias2.mul_(10.).round_()

# x = (torch.randn(b, i) * 10).round()
# y = af1(x)

# print("\n" + "x:" + "-" * 40)
# print(x.shape)
# print(x)

# print("\n" + "w1:" + "-" * 40)
# print(af1.weight1.shape)
# print(af1.weight1)

# print("\n" + "b1:" + "-" * 40)
# print(af1.bias1.shape)
# print(af1.bias1)

# print("\n" + "w2:" + "-" * 40)
# print(af1.weight2.shape)
# print(af1.weight2)

# print("\n" + "b2:" + "-" * 40)
# print(af1.bias2.shape)
# print(af1.bias2)

# print("\n" + "y:" + "-" * 40)
# print(y.shape)
# print(y)

b = 128
i = 12000
h = 10

x = torch.randn(b, i)
af1 = AnyF1(i, h, reLU_out=False)
y = af1(x)
print(x.var(), y.var())
