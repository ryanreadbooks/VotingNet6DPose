"""
@ Author: ryanreadbooks
@ Time: 9/14/2020, 20:20
@ File name: voting_net.py
@ File description: implement the simple network model. Simple network has less output channels in the end.
"""

import torch
import torch.nn.modules as nn

from configs import constants
from .backbone import Backbone
from .residual import Residual


class MaskBranch(nn.Module):
    """
    This is the simple version of VotingNetMaskBranch, whose output is tensor with number of channel is 2
    Output tensor shape (2, h, w),  for background, one for the object
    """

    def __init__(self) -> None:
        super(MaskBranch, self).__init__()
        self.conv1 = Residual(256, 256)
        self.conv2 = Residual(256, 256)
        self.up16to8 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv3 = Residual(256, 128)
        self.up8to4 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv4 = Residual(128, 64)
        self.up4to2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv5 = Residual(64, 64)
        self.up2to1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv6 = Residual(64, 2)
        self.conv11 = nn.Conv2d(2, 2, kernel_size=1, stride=1)

    def forward(self, x2s, x4s, x8s, x16s, x) -> torch.Tensor:
        """
        forward pass, the following shapes ignore the batch size channel
        :param x2s: 1/2 subsampled map, shape (64, h/2, w/2)
        :param x4s: 1/4 subsampled map, shape (64, h/4, w/4)
        :param x8s: 1/8 subsampled map, shape (128, h/4, w/4)
        :param x16s: 1/16 subsampled map, shape (256, h/16, h/16)
        :param x: output of backbone, shape (256, h/16, w/16)
        :return: shape (2, h, w)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.up16to8(x + x16s)
        x = self.conv3(x)
        x = self.up8to4(x + x8s)
        x = self.conv4(x)
        x = self.up4to2(x + x4s)
        x = self.conv5(x)
        x = self.up2to1(x + x2s)
        x = self.conv6(x)  # shape (n, num_class + 1, h, w)
        x = self.conv11(x)
        return torch.sigmoid(x)  # we use sigmoid to generate the probalibity


class VectorBranch(nn.Module):
    """
    Simple version of VotingNetVectorBranch, whose output tensor with shape of (2 * num_keypoints, h, w)
    """

    def __init__(self):
        super(VectorBranch, self).__init__()
        output_channel = constants.NUM_KEYPOINT * 2

        self.conv1 = Residual(256, 256)
        self.conv2 = Residual(256, 256)
        self.up16to8 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.up8to4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv3 = Residual(256, 128)
        self.conv4 = Residual(128, 64)

        self.up4to2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv5 = Residual(64, 64)

        self.up2to1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv6 = Residual(64, output_channel)

        self.conv11 = nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1)

    def forward(self, x2s, x4s, x8s, x16s, x):
        """
        forward pass, the following shapes ignore the batch size channel
        :param x2s: 1/2 subsampled map, shape (64, h/2, w/2)
        :param x4s: 1/4 subsampled map, shape (64, h/4, w/4)
        :param x8s: 1/8 subsampled map, shape (128, h/8, w/8)
        :param x16s: 1/16 subsampled map, shape (256, h/16, h/16)
        :param x: output of backbone, shape (256, h/16, w/16)
        :return:
        """
        x = self.conv1(x)
        x = self.conv2(x)  # shape (256, h/16, w/16)
        x = self.up16to8(x + x16s)  # shape (256, h/8, w/8)
        x = self.conv3(x)  # shape (128, h/8, w/8)
        x = x + x8s  # shape (128, h/8, w/8)
        x = self.up8to4(x)  # shape (128, h/4, w/4)
        x = self.conv4(x)  # shape (64, h/4, w/4)
        x = x + x4s  # shape (64, h/4, w/4)
        x = self.up4to2(x)  # shape (64, h/2, w/2)
        x = self.conv5(x)  # shape (64, h/2, w/2)
        x = x + x2s  # shape (64, h/2, w/2)
        x = self.up2to1(x)  # shape (64, h, w)
        x = self.conv6(x)  # shape (output_channel, h, w)
        x = self.conv11(x)
        return x


class VotingNet(nn.Module):
    """
    network thar integrate backbone, mask branch(with background output) and vector branch
    """

    def __init__(self, freeze_backbone: bool = True, freeze_mask_branch: bool = False, backbone_pretrained='ignore', local_path=None):
        """
        init function
        :param freeze_backbone: need to freeze the backbone or not, default = True
        :param freeze_mask_branch: need to freeze the mask branch or not, default = False
        :param backbone_pretrained: whether to use backbone pretrained model or not
        :param local_path: the given path on your machine
        """
        super(VotingNet, self).__init__()
        self.backbone = Backbone(freeze_backbone, backbone_pretrained, local_path)
        self.mask_branch = MaskBranch()
        if freeze_mask_branch:
            for p in self.mask_branch.parameters():
                p.requires_grad = False
        self.vector_branch = VectorBranch()

    def forward(self, x) -> (torch.Tensor, torch.Tensor):
        x2s, x4s, x8s, x16s, x = self.backbone(x)
        mask = self.mask_branch.forward(x2s, x4s, x8s, x16s, x)
        vector_map = self.vector_branch.forward(x2s, x4s, x8s, x16s, x)
        return mask, vector_map
