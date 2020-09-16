"""
@ Author: ryanreadbooks
@ Time: 9/14/2020, 20:32
@ File name: train_simple_net.py
@ File description: test the simple version of the network with less output channels
"""

import torch
import torchvision.transforms as torch_transform
import torch.utils.data as torch_data
import numpy as np

from nets import VotingNetSimple
from datasets import Linemod
from configs import constants
from configs import training_configs as cfgs
from trainer import Trainer


def main():
	# dataset
	img_transform = torch_transform.Compose([torch_transform.ToTensor(),
	                                         torch_transform.Normalize(mean=constants.IMAGE_MEAN, std=constants.IMAGE_STD)])
	linemod_dataset = Linemod(root_dir=constants.DATASET_PATH_ROOT, train=True,
	                          category='cat',
	                          transform=img_transform,
	                          simple=True)
	batch_size = cfgs.TRAINING_CONFIGS['batch_size']
	linemod_dataloader = torch_data.DataLoader(linemod_dataset, batch_size, shuffle=True, pin_memory=True)

	# init the network
	network = VotingNetSimple(freeze_backbone=True, freeze_mask_branch=False,
	                          pretrained='remote')
	mask_loss_fn = torch.nn.CrossEntropyLoss()
	vector_loss_fn = torch.nn.SmoothL1Loss(reduction='sum')
	trainer = Trainer(network=network, mask_loss_fn=mask_loss_fn, vector_loss_fn=vector_loss_fn)
	trainer.train(linemod_dataloader, resume=False)


if __name__ == '__main__':
	main()
