from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from configs import constants


class VotingNetBackbone(nn.Module):
	"""
	the backbone of the whole network
	"""

	def __init__(self, freeze: bool = True, pretrained: str = 'ignore', local_path=None):
		"""
		init function
		:param freeze: need to freeze the backbone or not, default=False
		:param pretrained: need pretrained model or not
		:param local_path: the given path on your machine
		"""
		super(VotingNetBackbone, self).__init__()
		# backbone resnet18
		resnet = None
		if pretrained == 'ignore':
			resnet = resnet18(False)
		elif pretrained == 'local':
			# load the pretrained model params from local machine
			if local_path is not None:
				resnet = resnet18(False)
				pre_trained_params = torch.load(pretrained)
				resnet.load_state_dict(pre_trained_params)
			else:
				raise ValueError('when pretrained is set to local, you need to specify a local path on your machine')
		elif pretrained == 'remote':
			# load the pretrained model params provided by torchvision
			resnet = resnet18(True)
		else:
			raise ValueError('pretrained must be \'ignore\', \'local\', \'remote\'')
		# drop the fc and avgpooling and layer4, because we only need the input to be subsampled to 1/16 of the original size
		resnet.fc = None
		resnet.avgpool = None
		resnet.layer4 = None

		self.conv1 = resnet.conv1
		self.bn1 = resnet.bn1
		self.relu = resnet.relu
		self.maxpool = resnet.maxpool
		self.layer1 = resnet.layer1
		self.layer2 = resnet.layer2
		self.layer3 = resnet.layer3

		# by applying this statement, the requires_grad of parameters before it is set to False
		# after it will not be affected
		for p in self.parameters():
			p.requires_grad = not freeze

		self.conv_dilated1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=2, dilation=2)
		self.conv_dilated2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=2, dilation=2)

		for p in self.parameters():
			p.requires_grad = False

	def forward(self, x) -> Tuple:
		x = self.conv1(x)
		x = self.bn1(x)
		x2s = self.relu(x)  # go to output, channel = 64
		x = self.maxpool(x2s)
		x4s = self.layer1(x)  # go to output, channel = 64
		x8s = self.layer2(x4s)  # go to output, channel = 128
		x16s = self.layer3(x8s)  # go to output, channel = 256
		x = self.conv_dilated1(x16s)
		x = self.relu(x)
		x = self.conv_dilated2(x)  # channel = 256
		return x2s, x4s, x8s, x16s, x


class VotingNetMaskBranch(nn.Module):
	"""
	the mask generation branch
	"""

	def __init__(self) -> None:
		super(VotingNetMaskBranch, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=256),
			nn.LeakyReLU(0.1, inplace=True)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=256),
			nn.LeakyReLU(0.1, True)
		)
		self.up16to8 = nn.UpsamplingBilinear2d(scale_factor=2)

		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=128),
			nn.LeakyReLU(0.1, True)
		)
		self.up8to4 = nn.UpsamplingBilinear2d(scale_factor=2)

		self.conv4 = nn.Sequential(
			nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=64),
			nn.LeakyReLU(0.1, True)
		)
		self.up4to2 = nn.UpsamplingBilinear2d(scale_factor=2)

		self.conv5 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=64),
			nn.LeakyReLU(0.1, True)
		)
		self.up2to1 = nn.UpsamplingBilinear2d(scale_factor=2)

		self.conv6 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=constants.NUM_CLS_LINEMOD, kernel_size=1, stride=1, bias=False),
		)

	def forward(self, x2s, x4s, x8s, x16s, x) -> torch.Tensor:
		"""
		forward pass, the following shapes ignore the batch size channel
		:param x2s: 1/2 subsampled map, shape (64, h/2, w/2)
		:param x4s: 1/4 subsampled map, shape (64, h/4, w/4)
		:param x8s: 1/8 subsampled map, shape (128, h/4, w/4)
		:param x16s: 1/16 subsampled map, shape (256, h/16, h/16)
		:param x: output of backbone, shape (256, h/16, w/16)
		:return:
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
		x = self.conv6(x)  # shape (n, num_class, h, w)
		return F.softmax(x, dim=1)  # maybe output probability at every pixel?


class VotingNetMaskBranchWithBg(nn.Module):
	"""
	the mask generation branch which has background output as the last channel of the output tensor
	"""

	def __init__(self) -> None:
		super(VotingNetMaskBranchWithBg, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=256),
			nn.LeakyReLU(0.1, inplace=True)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=256),
			nn.LeakyReLU(0.1, True)
		)
		self.up16to8 = nn.UpsamplingBilinear2d(scale_factor=2)

		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=128),
			nn.LeakyReLU(0.1, True)
		)
		self.up8to4 = nn.UpsamplingBilinear2d(scale_factor=2)

		self.conv4 = nn.Sequential(
			nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=64),
			nn.LeakyReLU(0.1, True)
		)
		self.up4to2 = nn.UpsamplingBilinear2d(scale_factor=2)

		self.conv5 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=64),
			nn.LeakyReLU(0.1, True)
		)
		self.up2to1 = nn.UpsamplingBilinear2d(scale_factor=2)

		self.conv6 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=constants.NUM_CLS_LINEMOD + 1, kernel_size=1, stride=1, bias=True),
		)

	def forward(self, x2s, x4s, x8s, x16s, x) -> torch.Tensor:
		"""
		forward pass, the following shapes ignore the batch size channel
		:param x2s: 1/2 subsampled map, shape (64, h/2, w/2)
		:param x4s: 1/4 subsampled map, shape (64, h/4, w/4)
		:param x8s: 1/8 subsampled map, shape (128, h/4, w/4)
		:param x16s: 1/16 subsampled map, shape (256, h/16, h/16)
		:param x: output of backbone, shape (256, h/16, w/16)
		:return:
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
		return x


class VotingNetVectorBranch(nn.Module):
	"""
	the vector map generation branch that generates the vector map
	"""

	def __init__(self):
		super(VotingNetVectorBranch, self).__init__()
		output_channel = constants.NUM_CLS_LINEMOD * constants.NUM_KEYPOINT * 2

		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=256),
			nn.LeakyReLU(0.1, True)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=256),
			nn.LeakyReLU(0.1, True)
		)
		self.up16to8 = nn.UpsamplingBilinear2d(scale_factor=2)

		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=256),
			nn.LeakyReLU(0.1, True)
		)
		self.channel128to256 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1)
		self.up8to4 = nn.UpsamplingBilinear2d(scale_factor=2)

		self.conv4 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=256),
			nn.LeakyReLU(0.1, True)
		)
		self.channel64to256 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1)
		self.up4to2 = nn.UpsamplingBilinear2d(scale_factor=2)

		self.conv5 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(num_features=256),
			nn.LeakyReLU(0.1, True)
		)
		self.channel64to256_b = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1)
		self.up2to1 = nn.UpsamplingBilinear2d(scale_factor=2)

		self.conv6 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=output_channel, kernel_size=1, stride=1, bias=False),
			nn.BatchNorm2d(num_features=output_channel)
		)

	def forward(self, x2s, x4s, x8s, x16s, x):
		"""
		forward pass, the following shapes ignore the batch size channel
		:param x2s: 1/2 subsampled map, shape (64, h/2, w/2)
		:param x4s: 1/4 subsampled map, shape (64, h/4, w/4)
		:param x8s: 1/8 subsampled map, shape (128, h/4, w/4)
		:param x16s: 1/16 subsampled map, shape (256, h/16, h/16)
		:param x: output of backbone, shape (256, h/16, w/16)
		:return:
		"""
		x = self.conv1(x)
		x = self.conv2(x)  # shape (256, h/16, w/16)
		x = self.up16to8(x + x16s)  # shape (256, h/8, w/8)
		x = self.conv3(x)  # shape (256, h/8, w/8)
		x = x + self.channel128to256(x8s)  # shape (256, h/8, w/8)
		x = self.up8to4(x)  # shape (256, h/4, w/4)
		x = self.conv4(x)  # shape (256, h/4, w/4)
		x = x + self.channel64to256(x4s)  # shape (256, h/4, w/4)
		x = self.up4to2(x)  # shape (256, h/2, w/2)
		x = self.conv5(x)  # shape (256, h/2, w/2)
		x = x + self.channel64to256_b(x2s)  # shape (256, h/2, w/2)
		x = self.up2to1(x)  # shape (256, h, w)
		x = self.conv6(x)  # shape (output_channel, h, w)
		return x


class VotingNet(nn.Module):
	"""
	network thar integrate backbone, mask branch and vector branch
	"""

	def __init__(self, pretrained='ignore', local_path=None):
		"""
		init function
		:param pretrained: whether to use pretrained model or not
		:param local_path: the given path on your machine
		"""
		super(VotingNet, self).__init__()
		self.backbone = VotingNetBackbone(True, pretrained, local_path)
		self.mask_branch = VotingNetMaskBranch()
		self.vector_branch = VotingNetVectorBranch()

	def forward(self, x) -> (torch.Tensor, torch.Tensor):
		x2s, x4s, x8s, x16s, x = self.backbone(x)
		mask = self.mask_branch(x2s, x4s, x8s, x16s, x)
		vector_map = self.vector_branch(x2s, x4s, x8s, x16s, x)
		return mask, vector_map


class VotingNetWithBg(nn.Module):
	"""
	network thar integrate backbone, mask branch(with background output) and vector branch
	"""

	def __init__(self, freeze_backbone: bool = True, freeze_mask_branch: bool = False, pretrained='ignore', local_path=None):
		"""
		init function
		:param freeze_backbone: need to freeze the backbone or not, default = True
		:param freeze_mask_branch: need to freeze the mask branch or not, default = False
		:param pretrained: whether to use pretrained model or not
		:param local_path: the given path on your machine
		"""
		super(VotingNetWithBg, self).__init__()
		self.backbone = VotingNetBackbone(freeze_backbone, pretrained, local_path)
		self.mask_branch = VotingNetMaskBranchWithBg()
		if freeze_mask_branch:
			for p in self.mask_branch.parameters():
				p.requires_grad = False
		self.vector_branch = VotingNetVectorBranch()

	def forward(self, x) -> (torch.Tensor, torch.Tensor):
		x2s, x4s, x8s, x16s, x = self.backbone(x)
		mask = self.mask_branch(x2s, x4s, x8s, x16s, x)
		vector_map = self.vector_branch(x2s, x4s, x8s, x16s, x)
		return mask, vector_map


# module testing
if __name__ == '__main__':
	inx = torch.rand(1, 3, 240, 320).to('cuda')
	vn = VotingNet().to('cuda')
	params = [p for p in vn.parameters() if p.requires_grad]
	print(params)
	for i in vn.parameters():
		# print(i)
		print(i.requires_grad)
