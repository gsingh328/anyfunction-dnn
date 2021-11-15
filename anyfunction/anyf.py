import math
import pathlib
import os

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Module, init

import numpy as np
import matplotlib.pyplot as plt


class AnyF1(Module):
    def __init__(self, input_shape, hidden_features, bias: bool = True,
                actv_out: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AnyF1, self).__init__()

        self.input_shape = list(input_shape)
        self.hidden_features = hidden_features
        self.bias = True
        self.actv_out = actv_out

        # The shape of weights is same as input
        # but the batch dimension is 1
        # and we add an extra dim at the end size of hidden_features (ie the expansion)
        self.weight_shape = [1] + self.input_shape + [hidden_features]

        self.weight1 = Parameter(torch.empty(self.weight_shape, **factory_kwargs))
        self.weight2 = Parameter(torch.empty(self.weight_shape, **factory_kwargs))
        if bias:
            self.bias1 = Parameter(torch.empty(self.weight_shape, **factory_kwargs))
            self.bias2 = Parameter(torch.empty(self.input_shape, **factory_kwargs))
        else:
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # gain = 1 if self.actv_out else 1.0
        # bound = 5
        # init.uniform_(self.weight1, -bound, bound)
        # bound = math.sqrt(3./self.hidden_features)
        # init.uniform_(self.weight2, -bound, bound)
        # init.normal_(self.weight2, std=math.sqrt(gain / self.hidden_features))

        # with torch.no_grad():
            # self.weight1.zero_()
            # self.weight2.zero_()

        init.normal_(self.weight2, 0, .2)
        init.normal_(self.weight1, 0, .2)

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
        pre_x = input.clone()

        x_shape = [-1] + self.input_shape + [1]
        x = input.view(x_shape)

        # Expansion
        x = x.mul(self.weight1)
        if self.bias1 is not None:
            x += self.bias1
        x = F.hardtanh(x)

        # Merge
        x = x.mul(self.weight2).sum(dim=-1)
        if self.bias2 is not None:
            x += self.bias2
        if self.actv_out:
            x = F.hardtanh(x)

        x = x + pre_x
        return x

    def debug_print_graph(self, log_folder="./debug/graphs", prefix=""):
        log_path = pathlib.Path(log_folder)
        log_path.mkdir(parents=True, exist_ok=True)

        nsamples = 1024
        x = torch.ones(nsamples, *self.input_shape).to(self.weight1.device)
        x = x * torch.linspace(-3, 3, nsamples).view(nsamples, *([1]*len(self.input_shape))).to(self.weight1.device)
        y = self.forward(x)

        x = x.view(nsamples, -1)
        y = y.view(nsamples, -1)

        ninputs = int(x.shape[1])
        for inp in range(ninputs):
            x_i = x[:, inp].cpu().detach().numpy()
            y_i = y[:, inp].cpu().detach().numpy()

            plt.plot(x_i, y_i)
            plt.title("AnyF for i: {}".format(inp))
            plt.xlabel("Input")
            plt.ylabel("Output")
            plt.savefig(os.path.join(log_folder, "{}plot_{}.png").format(prefix, inp))
            plt.clf()


class AnyF2(Module):
    def __init__(self, input_shape, hidden_features, bias: bool = True,
                actv_out: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AnyF2, self).__init__()

        self.input_shape = list(input_shape)
        self.hidden_features = hidden_features
        self.bias = True
        self.actv_out = actv_out

        # The shape of weights is same as input
        # but the batch dimension is 1
        # and we add an extra dim at the end size of hidden_features (ie the expansion)
        self.weight_shape = self.input_shape + [hidden_features]

        self.weight1 = Parameter(torch.empty(self.weight_shape, **factory_kwargs))
        self.weight2 = Parameter(torch.empty(self.weight_shape, **factory_kwargs))
        if bias:
            # self.register_buffer('bias1', torch.empty(self.weight_shape, **factory_kwargs))
            self.bias1 = Parameter(torch.empty(self.weight_shape, **factory_kwargs))
            # self.register_buffer('bias2', torch.empty(self.input_shape, **factory_kwargs))
            self.bias2 = Parameter(torch.empty(self.input_shape, **factory_kwargs))
        else:
            self.register_buffer('bias1', None)
            self.register_buffer('bias2', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # gain = 1 if self.actv_out else 1.0
        # bound = 5
        # init.uniform_(self.weight1, -bound, bound)
        # bound = math.sqrt(3./self.hidden_features)
        # init.uniform_(self.weight2, -bound, bound)
        # init.normal_(self.weight2, std=math.sqrt(gain / self.hidden_features))

        # with torch.no_grad():
            # self.weight1.zero_()
            # self.weight2.zero_()

        init.normal_(self.weight1, 0, .2)
        init.normal_(self.weight2, 0, .2)

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
        pre_x = input.clone()

        x_shape = list(input.shape) + [1]
        x = input.view(x_shape)

        # Expansion
        x = x.mul(self.weight1)
        if self.bias1 is not None:
            x += self.bias1
        x = F.gelu(x)

        # Merge
        x = x.mul(self.weight2).sum(dim=-1)
        if self.bias2 is not None:
            x += self.bias2
        if self.actv_out:
            x = F.gelu(x)

        x = x + pre_x
        return x

    def debug_print_graph(self, log_folder="./debug/graphs", prefix=""):
        log_path = pathlib.Path(log_folder)
        log_path.mkdir(parents=True, exist_ok=True)

        nsamples = 1024
        x = torch.ones(nsamples, *self.input_shape).to(self.weight1.device)
        x = x * torch.linspace(-3, 3, nsamples).view(nsamples, *([1]*len(self.input_shape))).to(self.weight1.device)
        y = self.forward(x)

        x = x.view(nsamples, -1)
        y = y.view(nsamples, -1)

        ninputs = int(x.shape[1])
        for inp in range(ninputs):
            x_i = x[:, inp].cpu().detach().numpy()
            y_i = y[:, inp].cpu().detach().numpy()

            plt.plot(x_i, y_i)
            plt.title("AnyF for i: {}".format(inp))
            plt.xlabel("Input")
            plt.ylabel("Output")
            plt.savefig(os.path.join(log_folder, "{}plot_{}.png").format(prefix, inp))
            plt.clf()


class GroupedTransform(Module):
    def __init__(self, group_count) -> None:
        super(GroupedTransform, self).__init__()
        self.group_count = group_count


    def forward(self, input: Tensor) -> Tensor:
        shape = torch.IntTensor(list(input.shape))
        nelems = torch.prod(shape[1:])

        new_nelems = torch.ceil(nelems / self.group_count) * self.group_count

        x = torch.flatten(input, 1)
        filler = torch.zeros(x.shape[0], int(new_nelems) - int(nelems)).to(input.device)

        y = torch.cat((x, filler), 1).reshape(shape[0], -1, self.group_count)
        return torch.swapaxes(y, 0, 1)



class PositionalEncoding(Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        assert (x.shape[0] < self.max_len)
        x = x + self.pe[:x.size(0)]
        return x
