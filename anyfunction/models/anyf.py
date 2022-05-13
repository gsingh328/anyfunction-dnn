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

# For normal matrices
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

        x = torch.cat((x, filler), 1).reshape(shape[0], -1, self.group_count)
        return x

# For 2D images. The transformation occurs channel-wise, so the grouping will fold channels
class GroupedTransform2D(Module):
    def __init__(self, group_count) -> None:
        super(GroupedTransform2D, self).__init__()
        self.group_count = group_count


    def forward(self, input: Tensor) -> Tensor:
        shape = torch.IntTensor(list(input.shape))
        nchannels = torch.prod(shape[1])

        new_nchannels = torch.ceil(nchannels / self.group_count) * self.group_count

        x = input
        if new_nchannels > nchannels:
            filler = torch.zeros(x.shape[0], int(new_nchannels) - int(nchannels), shape[2], shape[3]).to(input.device)
            x = torch.cat((x, filler), 1)

        x = x.reshape(shape[0], int(new_nchannels/self.group_count), int(self.group_count), shape[2], shape[3])
        return x


class PositionalEncoding(Module):
    def __init__(self, d_model: int, max_len: int = 5000, pe_multiplier: float = 1., base_freq: float = 10000.):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.pe_multiplier = pe_multiplier

        if d_model % 2 == 1:
            d_model += 1

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(base_freq) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

        self.in_max_len = 0

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        assert x.shape[1] < self.max_len, "{} must be greater than {}".format(x.shape[1], self.max_len)
        assert x.shape[2] <= self.pe.shape[2], "{} must be less than or equal to {}".format(x.shape[2], self.pe.shape[2])

        # current_len = x.shape[1]
        # if self.in_max_len < current_len:
        #     self.in_max_len = current_len
        #     print(current_len)
        #     print(x.shape[2])

        x = (x + self.pe_multiplier * self.pe[0, :x.size(1), :x.size(2)]) / (1. + self.pe_multiplier)
        return x


class PositionalEncoding2D(PositionalEncoding):
    def __init__(self, *args, **kwargs):
        super(PositionalEncoding2D, self).__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, channel, width, height]
        """
        x_org_shape = x.shape

        """
        Positional offset applied to channel, ie each pixel in width x height image has the same offset.
        """
        # Move the width and height adjacent to batch_dim and merge
        x = x.swapaxes(1, 3)
        x = x.swapaxes(2, 4)

        x_shape = x.shape
        x = x.reshape(x_shape[0] * x_shape[1] * x_shape[2], x_shape[3], x_shape[4])
        x = super().forward(x)

        # Undo merge + swap
        x = x.reshape(*x_shape)

        x = x.swapaxes(1, 3)
        x = x.swapaxes(2, 4)

        """
        Positional offset applied to image, ie the image per channel is flattened as offset applied.
        So channelwise the same offset exists
        """
        """
        # Move the channel dim adjacent to batch_dim and merge
        x = x.swapaxes(1, 2)

        x_shape = x.shape
        x = x.reshape(x_shape[0] * x_shape[1], x_shape[2], x_shape[3] * x_shape[4])
        x = super().forward(x)

        # Undo merge + swap
        x = x.reshape(*x_shape)

        x = x.swapaxes(1, 2)
        """
        """
        Positional offset applied to image + channel.
        So channel + image is flattened and then offset is applied.
        """

        # x = x.reshape(x_org_shape[0], x_org_shape[1] * x_org_shape[2], x_org_shape[3] * x_org_shape[4])
        # x = super().forward(x)

        x = x.reshape(x_org_shape)
        return x


class BinaryPositionalEncoding(Module):
    def __init__(self, d_model: int, max_len: int = 5000, pe_multiplier: float = 1.):
        super(BinaryPositionalEncoding, self).__init__()
        self.max_len = max_len
        self.pe_multiplier = pe_multiplier

        # Binary Vector representation
        bit_mask = 2**torch.arange(d_model).byte()
        pe = torch.arange(1, max_len).byte().unsqueeze(-1).bitwise_and(bit_mask).ne(0).float()

        # Normalise between -1 and 1
        pe = 2.* pe - 1

        # Add the batch dimension. All batches will have some positional offset
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

        self.in_max_len = 0
 
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        assert x.shape[1] < self.max_len, "{} must be greater than {}".format(x.shape[1], self.max_len)
        assert x.shape[2] <= self.pe.shape[2], "{} must be less than or equal to {}".format(x.shape[2], self.pe.shape[2])

        # current_len = x.shape[1]
        # if self.in_max_len < current_len:
        #     self.in_max_len = current_len
        #     print(current_len)
        #     print(x.shape[2])

        x = (x + self.pe_multiplier * self.pe[0, :x.size(1), :x.size(2)]) / (1. + self.pe_multiplier)
        return x


class UniformRandomPositionalEncoding(Module):
    def __init__(self, d_model: int, max_len: int = 5000, pe_multiplier: float = 1.):
        super(UniformRandomPositionalEncoding, self).__init__()
        self.max_len = max_len
        self.pe_multiplier = pe_multiplier

        pe = torch.rand(int(max_len), int(d_model))

        # Normalise between -1 and 1
        pe = 2.* pe - 1

        # Add the batch dimension. All batches will have some positional offset
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

        self.in_max_len = 0
 
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        assert x.shape[1] < self.max_len, "{} must be greater than {}".format(x.shape[1], self.max_len)
        assert x.shape[2] <= self.pe.shape[2], "{} must be less than or equal to {}".format(x.shape[2], self.pe.shape[2])

        # current_len = x.shape[1]
        # if self.in_max_len < current_len:
        #     self.in_max_len = current_len
        #     print(current_len)
        #     print(x.shape[2])

        x = (x + self.pe_multiplier * self.pe[0, :x.size(1), :x.size(2)]) / (1. + self.pe_multiplier)
        return x


class UniformPositionalEncoding(Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(UniformPositionalEncoding, self).__init__()
        self.max_len = max_len

        # pe = torch.rand(int(max_len), int(d_model))

        # Normalise between -1 and 1
        # pe = 2.* pe - 1

        ranges = torch.linspace(-1, 1, max_len * 2 + 1)

        pe = ranges[1::2].repeat(d_model, 1).T
        min_range = ranges[0:-1:2].repeat(d_model, 1).T
        max_range = ranges[2::2].repeat(d_model, 1).T

        for i in range(d_model):
            indices = torch.randperm(pe.shape[0])
            pe[:,i] = pe[indices, i]
            min_range[:,i] = min_range[indices, i]
            max_range[:,i] = max_range[indices, i]

        # Add the batch dimension. All batches will have some positional offset
        pe = pe.unsqueeze(0)
        min_range = min_range.unsqueeze(0)
        max_range = max_range.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        self.register_buffer('min_range', min_range)
        self.register_buffer('max_range', max_range)

        self.in_max_len = 0
 
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        assert x.shape[1] < self.max_len, "{} must be greater than {}".format(x.shape[1], self.max_len)
        assert x.shape[2] <= self.pe.shape[2], "{} must be less than or equal to {}".format(x.shape[2], self.pe.shape[2])

        # current_len = x.shape[1]
        # if self.in_max_len < current_len:
        #     self.in_max_len = current_len
        #     print(current_len)
        #     print(x.shape[2])

        x = x + self.pe[0, :x.size(1), :x.size(2)]
        return x


if __name__ == "__main__":
    x = UniformPositionalEncoding(8, 4)
    print(x.pe[0, :, 0])
    print(x.min_range[0, :, 0])
    print(x.max_range[0, :, 0])
    # print(x.pe)

