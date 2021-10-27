import math

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Module, init


class AnyF1(Module):
    def __init__(self, in_features, hidden_features, bias: bool = True,
                reLU_out: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AnyF1, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.bias = True
        self.reLU_out = reLU_out

        self.weight1 = Parameter(torch.empty((in_features, hidden_features), **factory_kwargs))
        self.weight2 = Parameter(torch.empty((in_features, hidden_features), **factory_kwargs))
        if bias:
            self.bias1 = Parameter(torch.empty((in_features, hidden_features), **factory_kwargs))
            self.bias2 = Parameter(torch.empty((in_features), **factory_kwargs))
        else:
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = 1 if self.reLU_out else 1.0
        bound = 5
        init.uniform_(self.weight1, -bound, bound)
        bound = math.sqrt(3./self.hidden_features)
        init.uniform_(self.weight2, -bound, bound)
        # init.normal_(self.weight2, std=math.sqrt(gain / self.hidden_features))

        # Keep bias initialization simple since the weights are random, we don't need
        # bias also to be random
        if self.bias1 is not None:
            with torch.no_grad():
                init.uniform_(self.bias1, -1, 1)
                # self.bias1.fill_(0.)
        if self.bias2 is not None:
            with torch.no_grad():
                init.uniform_(self.bias2, -1, 1)
                # self.bias1.fill_(0.)

    def forward(self, input: Tensor) -> Tensor:
        assert len(input.shape) == 2, "Invalid shape for input: {}".format(input.shape)

        input_shape = input.shape
        x = input.view(-1, self.in_features, 1)

        # Expansion-1
        x = x.mul(self.weight1.view(1, self.in_features, self.hidden_features))
        if self.bias1 is not None:
            x += self.bias1.view(1, self.in_features, self.hidden_features)
        x = F.hardtanh(x)

        # Expansion-2 + Merge
        x = x.mul(self.weight2.view(1, self.in_features, self.hidden_features)).sum(dim=-1)
        if self.bias2 is not None:
            x += self.bias2.view(1, self.in_features)
        if self.reLU_out:
            x = F.hardtanh(x, min_val=0.)
        return x
