import torch
from anyf import AnyF1


# b = 128
# i = 12000
# h = 10

# x = torch.randn(b, i)
# af1 = AnyF1(i, h, actv_out=False)
# y = af1(x)
# print(x.var(), y.var())

b = 128
h = 20
i_c = 1
i_h = 7
i_w = 7

x = (torch.randn(b, i_c, i_h, i_w) * 10).floor_()
# print("\n" + "x:" + "-" * 40)
# print(x.shape)
# print(x)

# w1 = (torch.randn(1, i_c, i_h, i_w, h) * 10).floor_()
# print("\n" + "w1:" + "-" * 40)
# print(w1.shape)
# print(w1)

# y1 = x.view(-1, i_c, i_h, i_w, 1).mul(w1)
# print("\n" + "y1:" + "-" * 40)
# print(y1.shape)
# print(y1)

af1 = AnyF1((i_c, i_h, i_w), h, actv_out=False)
af1.debug_print_graph()
