"""
@ Author: ryanreadbooks
@ Time: 2021/1/12
@ File name: residual.py
@ File description: A resblock like module
"""
import torch.nn.modules as nn


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        """
        residual block
        :param inp_dim: input dimension
        :param out_dim: output dimension
        """
        super(Residual, self).__init__()
        # the channel must be at least 1
        out_dim_half = max(1, int(out_dim / 2))

        self.conv1 = nn.Conv2d(inp_dim, out_dim_half, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim_half)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_dim_half, out_dim_half, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim_half)

        self.conv3 = nn.Conv2d(out_dim_half, out_dim, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim)

        self.skip_layer = nn.Conv2d(inp_dim, out_dim, 1, 1, bias=False)
        self.skip_layer_bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        residual = self.skip_layer(x)
        residual = self.skip_layer_bn(residual)  # skip connection output

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        self.relu(out)

        return out
