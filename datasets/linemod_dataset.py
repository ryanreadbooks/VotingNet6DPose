from typing import Tuple, List
import random
import os

import numpy as np
import torch
import torch.utils.data as torch_utils_data
from PIL import Image
import cv2

from configs import constants, training_configs
from utils.geometry_utils import get_model_corners
from utils.io_utils.inout import load_model_points
from utils.io_utils import load_depth_image


class Linemod(torch_utils_data.Dataset):
	"""
	LINEMOD dataset in my format, including mask for object in a image, the unit vector field for the image.
	The root path of the dataset folder in your local machine should be organized in the following structure:
		root:
			ape:
				JPEGImages: all the images of this object are in this folder
				labels: all labels of the object in images are here
				mask: all masks of the object in images are here
				gt_poses: all the ground truth poses are in this folder
				depth: all depth images are in this folder
				train.txt: the images(masks/labels) that are used for training
				test.txt: the images(masks/labels) that are used for testing
				ape.ply: the object model
				ape.xyz: the object point cloud
				...
			benchvise:
				...
			cam:
				...
			can:
				...
			...
	Attention: make sure the listed sub-folders and files are in one single object folder
	"""

	_legal_categories = ['all', 'ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck',
	                     'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']

	# todo: consider removing the dataset_size argument and use all training data
	def __init__(self, root_dir, train=True, category: str = 'all', dataset_size: int = None,
	             transform=None, need_bg: bool = False,
	             onehot: bool = True,
	             simple: bool = False):
		"""
		init function
		:param root_dir: root path of the dataset on your local machine
		:param train: return data or not, default=True
		:param category: what object's data you need, you can specify a specific object, like cat
		:param dataset_size: the number of data to return, default=780 for training and default=130 for testing
		:param transform: the transform you want to apply on the image
		:param need_bg: need the mask label to contain the background class or not, default= False
		:param onehot: need the mask label to be onehot format or not, default=True
		:param simple: need the simple version of target or not, this is designed for the simple version of network, default=False
		"""
		super(Linemod, self).__init__()
		self.root_dir = root_dir
		self.dir_container = dict()
		self.data_size = dataset_size
		self.data_img_path = list()  # list that stores JPEGImages's path
		self.data_mask_path = list()  # list that stores mask image's path
		self.data_labels_path = list()  # list that stores labels's path
		self.transform = transform
		self.need_bg = need_bg
		self.onehot = onehot
		self.simple = simple
		if dataset_size is None:
			# set the default datasize for training and testing
			if train:
				self.data_size = 60 * constants.NUM_CLS_LINEMOD
			else:
				self.data_size = 10 * constants.NUM_CLS_LINEMOD

		train_or_test = 'train.txt'
		if not train:
			train_or_test = 'test.txt'
		if category not in self._legal_categories:
			raise ValueError('invalid value for category')
		elif category == 'all':
			object_names = constants.LINEMOD_OBJECTS_NAME
		else:
			object_names = [category]
		# fetch all file(train or test) of all objects
		for cls_name in object_names:
			cls_data_path = os.path.join(root_dir, cls_name, train_or_test)
			with open(cls_data_path, 'r') as f:
				lines = f.readlines()
				cls_data: List = [line.rstrip()[-10: -4] for line in lines]
				self.dir_container[cls_name] = cls_data  # just like {'ape': ['000000', '000001', ...], 'benchvise': ['000000', '000001', ...]}

		# randomly pick data_size data to form the dataset
		each = self.data_size // constants.NUM_CLS_LINEMOD  # number of data for each class if use 'all'
		for i, item in enumerate(self.dir_container.items()):
			key, value = item
			# need to change the self.data_size if only one object's data is needed
			if category != 'all' and train:
				# if we specify only one class object, we use all the training images of that class object
				self.data_size = len(value)
				each = len(value)
			elif category != 'all' and not train:
				each = self.data_size

			if i == constants.NUM_CLS_LINEMOD - 1:
				each = self.data_size - each * i  # the rest
			# randomly pick each data
			random.shuffle(value)
			sampled = random.sample(value, each)
			base_dir = os.path.join(root_dir, key)
			labels = 'labels'
			if training_configs.KEYPOINT_TYPE == 'fps':
				labels = 'labels_fps'
			for s in sampled:
				data_img_path = os.path.join(base_dir, 'JPEGImages', s + '.jpg')
				data_mask_path = os.path.join(base_dir, 'mask', s[-4:] + '.png')
				data_label_path = os.path.join(base_dir, labels, s + '.txt')
				self.data_img_path.append(data_img_path)
				self.data_mask_path.append(data_mask_path)
				self.data_labels_path.append(data_label_path)
		print('datasize: ', self.data_size)

	def __getitem__(self, index: int) -> Tuple:  # color, mask, vector maps, cls_label, color_image_path
		"""
		Get one data from the dataset
		:param index: index of the dataset
		:return: image itself, mask tensor, coordinate vector, cls_label, color_image_path
		"""
		data_img_path: str = self.data_img_path[index]
		data_mask_path: str = self.data_mask_path[index]
		data_label_path: str = self.data_labels_path[index]
		# retrieve relevant information
		cls_label, vector_maps = LinemodDatasetProvider.provide_keypoints_vector_maps(data_label_path, self.simple)
		color: Image = LinemodDatasetProvider.provide_image(data_img_path)
		if not self.simple:
			# the full version of output
			if not self.need_bg:
				# without background
				# shape(n_classes, h, w)
				mask: torch.Tensor = LinemodDatasetProvider.provide_mask(cls_label, data_mask_path)
			else:
				# with background
				if self.onehot:
					# with bg and needs onehot
					# shape (n_classes + 1, h, w)
					mask: torch.Tensor = LinemodDatasetProvider.provide_mask_with_bg_onehot(cls_label, data_mask_path)
				else:
					# with bg but doesn't need onehot
					# shape (h, w)
					mask: torch.Tensor = LinemodDatasetProvider.provide_mask_with_bg(cls_label, data_mask_path)
		else:
			# the simple version of output
			# shape (h, w)
			mask: torch.Tensor = LinemodDatasetProvider.provide_mask_with_bg_simple(data_mask_path)

		# transform the image if needed
		if self.transform is not None:
			color = self.transform(color)
		return color, mask, vector_maps, cls_label, data_img_path

	def __len__(self) -> int:
		return self.data_size


class LinemodDatasetProvider(object):
	"""
	Provide the corresponding data label (mask, coordinates of keypoints) according to the given path
	"""

	@staticmethod
	def provide_image(path: str) -> Image:
		"""
		provide the image according to the given path
		:param path: given path
		:return: returned PIL Image
		"""
		return Image.open(path)

	@staticmethod
	def provide_mask(cls_label: int, path: str) -> torch.Tensor:
		"""
		provide the corresponding mask tensor of shape (num_classes, H, W) according to the given path
		:param cls_label: to which object class this mask belongs, in linemod cls_label~[0, 12]
		:param path: given path
		:return: array with shape (n_classes, h, w)
		"""
		# original shape (H,W,3), with the identical value in 3 channels
		# we take the first channel
		mask_single = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
		# make corresponding area to be probability 1
		mask_single[mask_single == 255] = 1.0
		h, w = mask_single.shape
		# create a space for the label
		# shape (num_of_keypoints, h, w)
		mask_tensor = np.zeros((constants.NUM_CLS_LINEMOD, h, w), dtype=np.float32)
		mask_tensor[cls_label] = mask_single

		return torch.from_numpy(mask_tensor)

	@staticmethod
	def provide_mask_with_bg_onehot(cls_label: int, path: str) -> torch.Tensor:
		"""
		provide the mask with background class at the last channel of the output tensor, the mask is onehot format
		:param cls_label: given class label
		:param path: given path
		:return: array with shape (n_classes + 1, h, w)
		"""
		mask_single = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
		mask_single[mask_single == 255] = 1.0
		h, w = mask_single.shape
		mask_tensor = np.zeros((constants.NUM_CLS_LINEMOD + 1, h, w), dtype=np.float32)
		mask_tensor[cls_label] = mask_single
		mask_tensor[-1][mask_single == 0] = 1.0

		return torch.from_numpy(mask_tensor)

	@staticmethod
	def provide_mask_with_bg(cls_label: int, path: str) -> torch.Tensor:
		"""
		provide the mask with background, not onehot format
		:param cls_label: given class label
		:param path: given path
		:return: returned mask, array with shape (h, w), each pixel represents a class label range from [0 ~ n_classes+1]
		"""
		mask_single = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
		mask = np.where(mask_single == 255, cls_label, constants.NUM_CLS_LINEMOD)  # 13 -> background. shape (h, w)

		return torch.from_numpy(mask)

	@staticmethod
	def provide_mask_with_bg_simple(path: str) -> torch.Tensor:
		"""
		Provide the simple mask for the simple network with less output channels, the format of the target is not onehot
		:param path: given path
		:return: returned simple mask
		"""
		mask_single = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
		# shape (h, w)
		mask = np.where(mask_single == 255, 0, 1)  # we only use 1 and 0 for the mask, 0 for object, 1 for the background

		return torch.from_numpy(mask)

	@staticmethod
	def provide_keypoints_coordinates(path: str) -> Tuple[int, torch.Tensor]:
		"""
		Provide the keypoints coordinates
		:param path: given path
		:return: cls label, from 0~12;
		         keypoints coordinates of format (x, y), shape (9, 2)
		"""
		with open(path, 'r') as f:
			line = f.readline().strip()
			labels = [i for i in map(lambda x: float(x), line.split(' '))]
			cls_label, kps = labels[0], labels[1:]  # contains class label and keypoints
			keypoints: torch.Tensor = torch.tensor(kps)
			# recover the raw coordinates
			keypoints[::2] *= constants.LINEMOD_IMG_WIDTH  # x coordinates
			keypoints[1::2] *= constants.LINEMOD_IMG_HEIGHT  # y coordinates
			keypoints: torch.Tensor = torch.stack([keypoints[::2], keypoints[1::2]]).T  # of shape (10, 2)
		return int(cls_label), keypoints[:-1]  # drop the last one that is not keypoint coordinate

	@staticmethod
	def provide_keypoints_vector_maps(path: str, simple: bool = False) -> Tuple[int, torch.Tensor]:
		"""
		Provide the keypoints maps, which is a tensor of shape (9 x 2 x num_classes, H, W)
		:param path: given path to read the keypoints coordinates and generate keypoints maps,
					which are unit vectors pointing from every pixel to the keypoint coordinates
		:param simple: whether to return the simple vector map or not. The simple version of it has less output channels.
		:return: class label of the corresponding keypoints;
				corresponding keypoints maps of shape (2 x n_keypoints x n_classes, H, W)
		"""
		# generating mesh for all pixels
		width: int = constants.LINEMOD_IMG_WIDTH
		height: int = constants.LINEMOD_IMG_HEIGHT
		num_of_keypoints: int = constants.NUM_KEYPOINT

		x = torch.linspace(0, width - 1, width)
		y = torch.linspace(0, height - 1, height)
		mesh = torch.meshgrid([y, x])
		# shape (2 * 9, H, W) with the first channel is x0-coordinate, the second channel is y0-coordinate
		pixel_coordinate = torch.cat([mesh[1][None], mesh[0][None]], dim=0).repeat(num_of_keypoints, 1, 1)
		# load the cls_label and keypoints
		cls_label, keypoints = LinemodDatasetProvider.provide_keypoints_coordinates(path=path)
		channel_per_cls = 2 * num_of_keypoints  # 2 x 9 = 18
		if not simple:
			channel: int = constants.NUM_CLS_LINEMOD * channel_per_cls  # 13 x 18 = 234
		else:
			cls_label = 0  # in simple version, we don't need the cls_label to be specific
			channel: int = channel_per_cls  # simple output for simple network, n_channels: 2 x num_keypoints
		# for unity of the simple and full version of target, we still prepare a space for the target
		vector_maps = torch.zeros((channel, height, width))
		keypoints = keypoints.reshape((num_of_keypoints * 2, 1, 1))
		dif = keypoints - pixel_coordinate
		# calculate the norm of dif vector at every pixel location
		# v = (k - p) / |k - p|
		dif_norm: torch.Tensor = dif.reshape((-1, 2, height, width))
		# print(dif_norm.shape)   # shape (n_kp, 2, h, w)
		dif_norm = torch.norm(dif_norm, dim=1)
		# print(dif_norm.shape)   # shape (n_kp, h, w)
		dif_norm: torch.Tensor = dif_norm.reshape((-1, 1, height, width))
		dif_norm: torch.Tensor = dif_norm.repeat(1, 1, 2, 1).reshape((-1, height, width))
		vector_map: torch.Tensor = dif / dif_norm  # shape (2 * 9, H, W)
		vector_maps[cls_label * channel_per_cls: (cls_label + 1) * channel_per_cls] = vector_map

		return cls_label, vector_maps

	@staticmethod
	def provide_rotation(path: str) -> np.ndarray:
		"""
		Load the rotation matrix from Linemod
		:param path: given path
		:return: loaded rotation matrix, shape (3, 3)
		"""
		with open(path, 'r') as f:
			lines = f.readlines()[1:]
			tmp_list = list(map(lambda x: x.strip().split(' '), lines))
			rot: np.ndarray = np.array(tmp_list, dtype=np.float32)
		return rot

	@staticmethod
	def provide_translation(path: str) -> np.ndarray:
		"""
		Load the translation vector from Linemod
		:param path: given path
		:return: loaded translation vector ,shape (3, 1)
		"""
		# it's the same
		return LinemodDatasetProvider.provide_rotation(path)

	@staticmethod
	def provide_pose(path: str) -> np.ndarray:
		"""
		Load the transformation matrix from Linemod, given the path
		Input path is just like: 'constants.DATASET_PATH_ROOT\\category\\JPEGImages\\000185.jpg'
		:param path: given path
		:return: transformation matrix, shape (3, 4)
		"""
		base_path, file_name = os.path.split(path)
		base_path = base_path.replace('JPEGImages', 'gt_poses')
		file_num = int(os.path.splitext(file_name)[0])
		rot_path = os.path.join(base_path, 'rot' + str(file_num) + '.rot')
		tra_path = os.path.join(base_path, 'tra' + str(file_num) + '.tra')

		rotation: np.ndarray = LinemodDatasetProvider.provide_rotation(rot_path)
		translation: np.ndarray = LinemodDatasetProvider.provide_rotation(tra_path)
		return np.hstack([rotation, translation])

	@staticmethod
	def provide_depth(path: str) -> np.ndarray:
		"""
		Load the depth image from given path of Linemod dataset,
		Input path is just like: 'constants.DATASET_PATH_ROOT\\category\\JPEGImages\\000185.jpg'
		:param path: given path
		:return: depth image array, shape with (h, w)
		"""
		base_path, file_name = os.path.split(path)
		base_path = base_path.replace('JPEGImages', 'depth')
		file_num = os.path.splitext(file_name)[0][2:]
		depth_path = os.path.join(base_path, file_num + '.png')
		return load_depth_image(depth_path)

	@staticmethod
	def provide_3d_keypoints(path: str):
		"""
		Get the 3d keypoints of a model
		:param path: given path
		:return: 3d keypoints
		"""
		points = load_model_points(path=path)
		kps: np.ndarray = get_model_corners(points)  # shape (8,3)
		return kps

	@staticmethod
	def provide_model_points(path: str) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Return the model points and model 3d keypoints
		:param path: given path
		:return: model points; model keypoints
		"""
		points: np.ndarray = load_model_points(path=path).astype(np.float64)
		# points: np.ndarray = get_ply_model(path=path)
		kps: np.ndarray = get_model_corners(points).astype(np.float64)  # shape (8,3)
		return points, kps

# Module testing
# if __name__ == '__main__':
# 	import torchvision.transforms as transforms
#
# 	t = transforms.Compose([transforms.ToTensor(),
# 	                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
# 	                                             std=[0.229, 0.224, 0.225], inplace=True)])
# 	dataset = Linemod(r'E:\1Downloaded\datasets\LINEMOD_from_yolo-6d', False, 'cat', 4, transform=t, need_bg=True, onehot=False)
# 	dataloader = torch_utils_data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
#
# 	for j, data in enumerate(dataloader):
# 		print("index: ", j)
# 		colorimg, maskimg, coormap, cls_target, color_img_path = data
# 		print(f'color shape: {colorimg.shape}')
# 		print(f'mask shape: {maskimg.shape}')
# 		print(f'coormap shape: {coormap.shape}')
# 		print(f'class label: {cls_target}')
# 		print(color_img_path)
# 		print(maskimg[0][cls_target[0]].sum())
