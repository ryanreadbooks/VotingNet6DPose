"""
@ Author: ryanreadbooks
@ Time: 9/14/2020, 20:32
@ File name: train_net.py
@ File description: test the simple version of the network with less output channels
"""

import torch
import torch.nn.modules as nn
import torchvision.transforms as torch_transform
import torch.utils.data as torch_data

from nets import VotingNet
from datasets import Linemod
from configs import constants
from configs.configuration import train_config
from trainer import Trainer


def main():
    # dataset
    img_transform = torch_transform.Compose([torch_transform.ToTensor(),
                                             torch_transform.Normalize(mean=constants.IMAGE_MEAN, std=constants.IMAGE_STD)])
    linemod_dataset = Linemod(train=True, transform=img_transform)
    batch_size = train_config.batch_size
    linemod_dataloader = torch_data.DataLoader(linemod_dataset, batch_size, shuffle=True, pin_memory=True)
    # init the network
    network = VotingNet(freeze_backbone=True, freeze_mask_branch=False, backbone_pretrained='remote')
    mask_loss_fn = nn.BCELoss()
    vector_loss_fn = torch.nn.SmoothL1Loss(reduction='sum')
    trainer = Trainer(network=network, mask_loss_fn=mask_loss_fn, vector_loss_fn=vector_loss_fn)
    trainer.train(linemod_dataloader, resume=False)


if __name__ == '__main__':
    main()
    # net = VotingNet()
    # y = net(torch.rand(2, 3, 480, 640))
    # print(y[0].shape)
    # print(y[1].shape)
    # net = VotingNet()
    # print(net.vector_branch.conv11.out_channels)
