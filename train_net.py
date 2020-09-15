"""
Train the whole VotingNet
"""
import torch.utils.data as torch_data
import torch.nn as nn
import torchvision.transforms as torch_transform
from nets import VotingNet, VotingNetWithBg
from datasets import Linemod
from configs import constants
from configs import training_configs as cfgs
from trainer import Trainer, CrossEntropy


def main():
	# prepare training data
	img_transform = torch_transform.Compose([torch_transform.ToTensor(),
	                                         torch_transform.Normalize(mean=constants.IMAGE_MEAN, std=constants.IMAGE_STD)])
	linemod_dataset = Linemod(root_dir=constants.DATASET_PATH_ROOT, train=True, category='cat', dataset_size=1040, transform=img_transform)
	batch_size = cfgs.TRAINING_CONFIGS['batch_size']
	linemod_dataloader = torch_data.DataLoader(linemod_dataset, batch_size, shuffle=True, pin_memory=True)

	voting_net_model = VotingNet('remote')

	trainer = Trainer(voting_net_model, CrossEntropy(), nn.SmoothL1Loss())
	trainer.train(linemod_dataloader, resume=True, path='/content/voting_net_6d/log_info/ok_parmas_59.pth')


def main_train_net_with_mask_bg():
	img_transform = torch_transform.Compose([torch_transform.ToTensor(),
	                                         torch_transform.Normalize(mean=constants.IMAGE_MEAN, std=constants.IMAGE_STD)])
	linemod_dataset = Linemod(root_dir=constants.DATASET_PATH_ROOT,
	                          train=True, category='cat',
	                          dataset_size=1040,
	                          transform=img_transform,
	                          need_bg=True,
	                          onehot=False)
	batch_size = cfgs.TRAINING_CONFIGS['batch_size']
	linemod_dataloader = torch_data.DataLoader(linemod_dataset, batch_size, shuffle=True, pin_memory=True)

	voting_net_bg_model = VotingNetWithBg(freeze_backbone=True, freeze_mask_branch=False, pretrained='remote')
	trainer: Trainer = Trainer(voting_net_bg_model, nn.CrossEntropyLoss(), nn.SmoothL1Loss(reduction='sum'))
	trainer.train(linemod_dataloader, resume=True, path='/content/voting_net_6d/log_info/finetune_cat_79_0907_ok.pth')


if __name__ == '__main__':
	# main()
	main_train_net_with_mask_bg()
