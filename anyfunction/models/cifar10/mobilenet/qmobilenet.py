import torch
import torch.nn as nn
import torch.nn.functional as F

from anyfunction.models.quant_modules import QuantAct, QuantConv2D, QuantLinear


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1, quant_mode='none'):
        super(Block, self).__init__()
        self.act_bit = 8
        self.conv_weight_bit = 8

        self.pre_conv1 = QuantAct(self.act_bit, quant_mode=quant_mode, per_channel=False)
        self.conv1 = QuantConv2D(self.conv_weight_bit, quant_mode=quant_mode, per_channel=False)
        self.conv1.set_param(nn.Conv2d(in_planes, in_planes, kernel_size=3,
            stride=stride, padding=1, groups=in_planes, bias=False))

        self.bn1 = nn.BatchNorm2d(in_planes)

        self.pre_conv2 = QuantAct(self.act_bit, quant_mode=quant_mode, per_channel=False)
        self.conv2 = QuantConv2D(self.conv_weight_bit, quant_mode=quant_mode, per_channel=False)
        self.conv2.set_param(nn.Conv2d(in_planes, out_planes, kernel_size=1,
            stride=1, padding=0, bias=False))

        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x, scaling_factor):
        x, scaling_factor = self.pre_conv1(x, scaling_factor)
        x, scaling_factor = self.conv1(x, scaling_factor)
        x = F.relu(self.bn1(x))

        x, scaling_factor = self.pre_conv2(x, scaling_factor)
        x, scaling_factor = self.conv2(x, scaling_factor)
        x = F.relu(self.bn2(x))
        return x, scaling_factor


class QMobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, *args, num_classes=10, quant_mode='none' , **kwargs):
        super(QMobileNet, self).__init__()
        self.act_bit = 8
        self.conv_weight_bit = 8
        self.linear_weight_bit = 8
        self.linear_bias_bit = 32

        self.pre_conv1 = QuantAct(self.act_bit, quant_mode=quant_mode, per_channel=False)
        self.conv1 = QuantConv2D(self.conv_weight_bit, quant_mode=quant_mode, per_channel=False)
        self.conv1.set_param(nn.Conv2d(3, 32, kernel_size=3,
            stride=1, padding=1, bias=False))

        self.bn1 = nn.BatchNorm2d(32)

        self.layers = self._make_layers(in_planes=32)

        self.pre_linear = QuantAct(self.act_bit, quant_mode=quant_mode, per_channel=False)
        self.linear = QuantLinear(self.linear_weight_bit, bias_bit=self.linear_bias_bit,
            quant_mode=quant_mode, per_channel=False)
        self.linear.set_param(nn.Linear(1024, num_classes))

    def _make_layers(self, in_planes):
        layers = nn.ModuleList()
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return layers

    def forward(self, x):
        x, scaling_factor = self.pre_conv1(x)
        x, scaling_factor = self.conv1(x, scaling_factor)
        x = F.relu(self.bn1(x))

        for block in self.layers:
            x, scaling_factor = block(x, scaling_factor)

        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)

        x, scaling_factor = self.pre_linear(x, scaling_factor)
        x, scaling_factor = self.linear(x, scaling_factor)
        return x


if __name__ == '__main__':
    net = QMobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()
