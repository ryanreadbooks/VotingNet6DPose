"""
@ Author: ryanreadbooks
@ Time: 9/7/2020, 19:13
@ File name: output_extractor.py
@ File description: Convenient extract the needed part of output tensor of the neural network
"""

import numpy as np
from configs import constants


class OutputExtractor(object):
	"""
	the class can extract some information of a specific class from the network output
	"""
	@staticmethod
	def extract_mask(out: np.ndarray, cls: int) -> np.ndarray:
		"""
		extract the specific mask according to the given input and class label
		:param out: the input array of shape (num_classes, h, w)
		:param cls: the mask of cls you want to extract from out
		:return: extracted mask array of shape (h, w)
		"""
		return out[cls]

	@staticmethod
	def extract_vector_field(out: np.ndarray, cls: int) -> np.ndarray:
		"""
		extract the specific vector map according to the given input and class label
		:param out:  the input array of shape (num_classes * 2 * num_keypoints, h, w)
		:param cls: the vector map of cls you want to extract from out
		:return: extracted vector map array of shape (2 * num_keypoints, h ,w)
		"""
		num_channel_per_class = constants.NUM_KEYPOINT * 2
		return out[cls * num_channel_per_class: (cls + 1) * num_channel_per_class]


class LinemodOutputExtractor:
	"""
	the class can extract some information of a specific class from the network output, given linemod dataset
	"""
	def __init__(self) -> None:
		super().__init__()

	@staticmethod
	def extract_mask_by_name(out: np.ndarray, name: str) -> np.ndarray:
		"""
		extract the specific mask according to the given input and class label in LINEMOD Dataset
		:param out: the input array of shape (num_classes, h, w)
		:param name: the mask of cls you want to extract from out
		:return: extracted vector map array of shape (2 * num_keypoints, h ,w)
		"""
		if name not in constants.LINEMOD_OBJECTS_NAME:
			raise ValueError('object {} is not in the list of linemod object'.format(name))
		cls_label = constants.LINEMOD_OBJECTS_NAME.index(name)
		return OutputExtractor.extract_mask(out, cls_label)

	@staticmethod
	def extract_vector_field_by_name(out: np.ndarray, name: str) -> np.ndarray:
		"""
		extract the specific vector map according to the given input and class label
		:param out: the input array of shape (num_classes * 2 * num_keypoints, h, w)
		:param name: the vector map of cls you want to extract from out
		:return: extracted vector map array of shape (2 * num_keypoints, h ,w)
		"""
		if name not in constants.LINEMOD_OBJECTS_NAME:
			raise ValueError('object {} is not in the list of linemod object, \ncheck {} for information'
			                 .format(name, constants.LINEMOD_OBJECTS_NAME))
		cls_label = constants.LINEMOD_OBJECTS_NAME.index(name)
		return OutputExtractor.extract_vector_field(out, cls_label)
