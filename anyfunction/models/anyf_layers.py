import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter

from .anyf import *
from .quant_modules import *


# Assumes the input is within the [-1, 1) range due to quantization
class FoldedAnyFLinear(Module):
    def __init__(self, in_features, out_features, bias=True, fold_factor=2,
            nbits=8, anyf=True, eps=1e-5):
        super(FoldedAnyFLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fold_factor = fold_factor
        self.nbits = nbits
        self.anyf = anyf
        self.eps = eps

        self.isize = int(in_features//fold_factor)
        self.osize = int(out_features//fold_factor)

        assert self.isize * fold_factor == in_features
        assert self.osize * fold_factor == out_features

        self.q = QuantAct(nbits, quant_mode='symmetric')

        self.layers = nn.ModuleList()
        for i in range(fold_factor):
            self.layers.append(nn.Linear(self.isize, self.osize, bias=False))

        self.p = UniformPositionalEncoding(in_features, fold_factor)

    def forward(self, input):
        x = input.view(input.shape[0], self.fold_factor, -1)

        x, s = self.q(x, clamp_only=True)
        n = 2 ** (self.nbits - 1) - 1
        s = s * n

        # Make x [-1,1)
        # x = x / s

        # Make x fall within one-segment of anyf
        # x = x / self.fold_factor

        pe_offset = self.p.pe[0, :x.shape[1], :x.shape[2]].reshape(1, x.shape[1], x.shape[2])
        x = x + pe_offset

        out = []
        for i in range(self.fold_factor):
            if self.anyf:
                # Have the folds merged in to batch dim
                x_fold = fold_idx_ste.apply(x, i)
                x_folded = x_fold.view(-1, x.shape[2])

                min_range = self.p.min_range[0, i, :x_folded.shape[1]].reshape(1, x_folded.shape[1])
                max_range = self.p.max_range[0, i, :x_folded.shape[1]].reshape(1, x_folded.shape[1])

                # Properly scale the segment boundaries of our anyf to the scale of quantized input
                min_range *= s
                max_range *= s

                # Decrease the ranges by eps to prevent float equavilanet issues
                # min_range += self.eps
                # max_range -= self.eps

                mask = torch.logical_or(x_folded < min_range, x_folded > max_range)

                # x_folded = x_folded.view(x.shape[0], self.fold_factor, -1)
                # x_folded = x_folded - pe_offset
                # x_folded = x_folded * self.fold_factor
                # x_folded = x_folded * s
                # x_folded = x_folded + pe_offset
                # x_folded = x_folded.view(-1, x.shape[2])

                # print('-' * 50)
                # print(min_range)
                # print(x_folded)
                # print(max_range)
                # print('-' * 50)
                # print(mask)

                # tmp = int((mask == False).sum())
                # assert tmp <= x.shape[0] * x.shape[2], '%s > %s' % (tmp, x.shape[0] * x.shape[2])
                # if tmp != x.shape[0] * x.shape[2]:
                    # print('mismatch')
                    # exit(1)

                x_folded_masked = x_folded.masked_fill(mask, 0.)

                # mask = torch.zeros_like(x)
                # mask[:, i, :] = 1.
                # x_folded_masked = (x * mask).view(-1, x.shape[2])
                # print('-' * 50)
                # print(x_folded_masked)

                # The output for i'th fold
                # Accumulate the results from other ranges to get this one
                out_fold_i = self.layers[i](x_folded_masked).view(x.shape[0], self.fold_factor, -1)
                # out_fold_i = fold_idx_ste.apply(out_fold_i, i)
                out_fold_i = out_fold_i.sum(dim=1)

                # print(out_fold_i)
                # print(self.layers[i](x[:, i, :]))
            else:
                out_fold_i = self.layers[i](x[:, i, :]) #.view(x.shape[0], self.fold_factor, -1)

            out.append(out_fold_i)

        x = torch.stack(out, dim=1)
        x = x.reshape(x.shape[0], -1)
        return x

    def debug_print_graph(self):
        weights = []
        for i in range(self.fold_factor):
            w = self.layers[i].weight.data
            w[w.abs() < 1/(2.**self.nbits)] = 0.
            weights.append(w)

        w = torch.stack(weights, dim=0).cpu().numpy()

        for i in range(w.shape[0]):
            x = w[i, :, :].flatten()
            x_z = (x == 0.).sum() / x.size

            ax = plt.gca()
            n, bins, patches = plt.hist(x, 2**(self.nbits-1))
            plt.xlabel('Weights')
            plt.ylabel('Counts')
            plt.text(.99, .99, '# Zero Ratio: %.2f' % float(x_z), transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='top')

            plt.savefig("graphs/hist_wz_i_%d.png" % i, dpi=400)
            plt.clf()

            x = x[x != 0.]

            ax = plt.gca()
            n, bins, patches = plt.hist(x, 2**(self.nbits-1))
            plt.xlabel('Weights')
            plt.ylabel('Counts')
            plt.text(.99, .99, '# Zero Ratio: %.2f' % float(x_z), transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='top')

            plt.savefig("graphs/hist_woz_i_%d.png" % i, dpi=400)
            plt.clf()

        w = w.reshape((self.fold_factor, -1))
        print(w.shape)
        w_nz_count = (np.abs(w) > 0).sum(axis=0)
        # w_nz_count = w_nz_count[w_nz_count > 0]
        print(w.T)
        print(w_nz_count.shape)

        y = np.bincount(w_nz_count)
        ii = np.nonzero(y)[0]
        print(np.vstack((ii,y[ii])).T)


        ax = plt.gca()
        n, bins, patches = plt.hist(w_nz_count, np.arange(w_nz_count.min(), w_nz_count.max()+2), align='left')
        plt.xlabel('Non-Zero Segments')
        plt.ylabel('Counts')
        plt.savefig("graphs/non_zero_segments.png", dpi=400)
        plt.clf()


class FoldedAnyFConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,\
            bias=False, fold_factor=2, nbits=8, anyf=True):
        super(FoldedAnyFConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fold_factor = fold_factor
        self.nbits = nbits
        self.anyf = anyf

        self.isize = int(in_channels//fold_factor)
        self.osize = int(out_channels//fold_factor)

        assert self.isize * fold_factor == in_channels
        assert self.osize * fold_factor == out_channels

        self.q = QuantAct(nbits, quant_mode='symmetric')

        self.layers = nn.ModuleList()
        for i in range(fold_factor):
            self.layers.append(nn.Conv2d(self.isize, self.osize, kernel_size, stride=stride,\
                padding=padding, bias=False))

        self.p = UniformPositionalEncoding(in_channels, fold_factor)


    def forward(self, input):
        folded_shape = (input.shape[0], self.fold_factor, -1, input.shape[2], input.shape[3])
        x = input.view(*folded_shape)

        x, s = self.q(x, clamp_only=True)
        n = 2 ** (self.nbits - 1) - 1
        s = s * n

        pe_offset = self.p.pe[0, :x.shape[1], :x.shape[2]].reshape(1, x.shape[1], x.shape[2], 1, 1)
        x = x + pe_offset

        out = []
        for i in range(self.fold_factor):
            if self.anyf:
                x_folded = x.clone().view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])

                min_range = self.p.min_range[0, i, :x_folded.shape[1]].reshape(1, x_folded.shape[1], 1, 1)
                max_range = self.p.max_range[0, i, :x_folded.shape[1]].reshape(1, x_folded.shape[1], 1, 1)

                # Properly scale the segment boundaries of our anyf to the scale of quantized input
                min_range *= s
                max_range *= s

                mask = torch.logical_or(x_folded < min_range, x_folded > max_range)
                x_folded_masked = x_folded.masked_fill(mask, 0.)

                out_fold_i = self.layers[i](x_folded_masked)
                out_fold_i = out_fold_i.view(input.shape[0], self.fold_factor, -1, out_fold_i.shape[2], out_fold_i.shape[3])
                # print(out_fold_i.shape)
                out_fold_i = out_fold_i.sum(dim=1)
                # print(out_fold_i.shape)
                # exit(1)
            else:
                out_fold_i = self.layers[i](x[:, i])

            out.append(out_fold_i)

        x = torch.stack(out, dim=1)
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        return x


class AnyFLinear(Module):
    def __init__(self, in_features, out_features, bias=True,
            n_seg=2, nbits=8, act_fn=F.relu):
        super(AnyFLinear, self).__init__()
        self.n_seg = n_seg
        self.nbits = nbits
        self.act_fn = act_fn

        self.layers = nn.ModuleList()
        for i in range(self.n_seg):
            self.layers.append(nn.Linear(in_features, out_features, bias=False))

        self.q = QuantAct(nbits, quant_mode='symmetric')

        ranges = torch.linspace(-1, 1, n_seg * 2 + 1)
        bias = ranges[1::2].view(1, -1, 1)
        self.register_buffer('bias', bias)


    def forward(self, input):
        x, s = self.q(input, clamp_only=True)

        # Get proper scale so bias is properly scaled to input dynamic range
        n = 2 ** (self.nbits - 1) - 1
        s = s * n

        out = []
        for i in range(self.n_seg):
            # out_seg_i = self.layers[i](self.act_fn(x + self.bias[:, i]))
            out_seg_i = self.layers[i](x)
            out.append(out_seg_i)

        x = torch.stack(out, dim=1)
        x = x + self.bias * s
        x = self.act_fn(x)
        x = x.sum(dim=1)
        return x


class AnyFConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,\
            bias=False, n_seg=4, nbits=8, act_fn=F.relu):
        super(AnyFConv2D, self).__init__()
        self.n_seg = n_seg
        self.nbits=nbits
        self.act_fn = act_fn

        self.layers = nn.ModuleList()
        for i in range(self.n_seg):
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,\
                padding=padding, bias=False))

        self.q = QuantAct(nbits, quant_mode='symmetric')

        ranges = torch.linspace(-1, 1, n_seg * 2 + 1)
        bias = ranges[1::2].view(1, -1, 1, 1, 1)
        self.register_buffer('bias', bias)


    def forward(self, input):
        x, s = self.q(input, clamp_only=True)

        # Get proper scale so bias is properly scaled to input dynamic range
        n = 2 ** (self.nbits - 1) - 1
        s = s * n

        out = []
        for i in range(self.n_seg):
            # out_seg_i = self.layers[i](self.act_fn(x + self.bias[:, i]))
            out_seg_i = self.layers[i](x)
            out.append(out_seg_i)

        x = torch.stack(out, dim=1)
        x = x + self.bias * s
        x = self.act_fn(x)
        x = x.sum(dim=1)
        return x


if __name__ == '__main__':
    torch.manual_seed(0)
    # l = FoldedAnyFLinear(16, 4, fold_factor=4)
    # l.to('cuda')

    # for i in range(1):
    #     x = Parameter(torch.randn(2, 16) * 2.0, requires_grad=True).cuda()
    #     x.retain_grad()
    #     y = l(x)
    #     z = y.sum()
    #     z.backward()
    #     print('=' * 100)
    #     print(x.grad)

    # print('-'*50)
    # print(x.data)
    # print('-'*50)
    # print(y.data)
    # print('-'*50)
    # print(x.grad)

    # x = Parameter(torch.randn(8), requires_grad=True).cuda()
    # x.retain_grad()

    # x_l = x.masked_fill(x > 0, 0.)
    # x_r = x.masked_fill(x < 0, 0.)
    # y = (x_l * 2) + (x_r * 3)

    # z = y.sum()
    # z.backward()

    # print('-'*50)
    # print(x.data)
    # print('-'*50)
    # print(x.grad)
