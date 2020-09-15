"""
@ Author: ryanreadbooks
@ Time: 9/7/2020, 19:15
@ File name: draw_utils.py
@ File description: Visualization functions are defined here
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw

from configs import LINEMOD_MASK_COLOR, VECTOR_FIELD_COLOR
from configs import constants


def draw_3d_bbox(img: Image, box: np.ndarray, color='green') -> Image:
	"""
	draw the given 3d bbox in given image
	:param img: given image
	:param box: given 3d bbox coordinates(x, y) in ndarray of shape(8, 2); order: corner1 ~ corner8
	:param color: the color the draw the box
	:return: modified image
	"""
	if color not in ['blue', 'green']:
		raise ValueError('color type of {} is not supported, enter green or blue'.format(color))
	drawer = ImageDraw.Draw(img)
	x1, y1 = box[0, 0], box[0, 1]
	x2, y2 = box[1, 0], box[1, 1]
	x3, y3 = box[2, 0], box[2, 1]
	x4, y4 = box[3, 0], box[3, 1]
	x5, y5 = box[4, 0], box[4, 1]
	x6, y6 = box[5, 0], box[5, 1]
	x7, y7 = box[6, 0], box[6, 1]
	x8, y8 = box[7, 0], box[7, 1]
	line_color = (0, 255, 0)
	if color is 'blue':
		line_color = (0, 0, 255)
	line_width = 3
	drawer.line(xy=[x1, y1, x2, y2], fill=line_color, width=line_width)
	drawer.line(xy=[x1, y1, x3, y3], fill=line_color, width=line_width)
	drawer.line(xy=[x1, y1, x5, y5], fill=line_color, width=line_width)
	drawer.line(xy=[x2, y2, x4, y4], fill=line_color, width=line_width)
	drawer.line(xy=[x2, y2, x6, y6], fill=line_color, width=line_width)
	drawer.line(xy=[x3, y3, x4, y4], fill=line_color, width=line_width)
	drawer.line(xy=[x3, y3, x7, y7], fill=line_color, width=line_width)
	drawer.line(xy=[x4, y4, x8, y8], fill=line_color, width=line_width)
	drawer.line(xy=[x5, y5, x6, y6], fill=line_color, width=line_width)
	drawer.line(xy=[x5, y5, x7, y7], fill=line_color, width=line_width)
	drawer.line(xy=[x6, y6, x8, y8], fill=line_color, width=line_width)
	drawer.line(xy=[x7, y7, x8, y8], fill=line_color, width=line_width)

	return img


def draw_points(img, points, need_text=True, color='green') -> Image:
	"""
	draw the given points in the given image
	:param img: given image
	:param points: given points of shape (n, 2) with n the number of points. order~(x, y)
	:param need_text: whether to draw the order of the points in the image as well
	:param color: the color of points
	:return: image with points and text drawn
	"""
	draw = ImageDraw.Draw(img)
	r = 3  # radius of point
	tsy = 30  # position of text
	point_color = (0, 255, 0)
	if color == 'blue':
		point_color = (0, 0, 255)
	for i, points in enumerate(points):
		x, y = points[0], points[1]
		if y == 0:
			tsy = -10
		if need_text:
			draw.text((x - r, y - tsy), str(i), fill=point_color)
		draw.ellipse((x - r, y - r, x + r, y + r), fill=point_color)
	return img


def draw_linemod_label(label: np.ndarray) -> Image:
	"""
	visualize the label of the image of every pixel
	:param label: the given mask, shape of (num_classes, h, w), linemod dataset has 13 classes, or 13 + 1(background) classes
	:return: output label rgb image without the mask overlapping on it
	"""
	h, w = label.shape[1], label.shape[2]
	label_img = np.zeros((3, h, w))  # all black
	for i, obj in enumerate(label):
		# obj shape (h, w)
		label_img[:, :, :] += obj[None]
		label_img[0][label_img[0] == 1] = LINEMOD_MASK_COLOR[i][0]
		label_img[1][label_img[1] == 1] = LINEMOD_MASK_COLOR[i][1]
		label_img[2][label_img[2] == 1] = LINEMOD_MASK_COLOR[i][2]
	label_img = label_img.astype(np.uint8)
	img_out = Image.fromarray(label_img.transpose([1, 2, 0]), 'RGB')
	return img_out


def draw_linemod_mask_v2(mask: np.ndarray) -> Image:
	onehot_mask = mask_onehot_converter(mask, constants.NUM_CLS_LINEMOD)
	return draw_linemod_label(onehot_mask)


def mask_onehot_converter(mask: np.ndarray, n_class: int):
	"""

	:param mask: mask label, but not onehot format, shape of (h, w)
	:param n_class: number of classes
	:return: onehot format label mask, shape of (n_classes, h, w)
	"""
	h, w = mask.shape[0], mask.shape[1]
	onehot = np.zeros((n_class, h, w), dtype=np.uint8)
	for cls in range(n_class):
		onehot[cls][mask == cls] = 1
	return onehot


def draw_linemod_mask(mask: np.ndarray) -> Image:
	"""
	visualize mask
	:param mask: mask array of shape (h, w)
	:return: image
	"""
	h, w = mask.shape[0], mask.shape[1]
	mask_img = np.zeros((3, h, w))  # all black
	for key, color in LINEMOD_MASK_COLOR.items():
		mask_img[0][mask == key] = color[0]
		mask_img[1][mask == key] = color[1]
		mask_img[2][mask == key] = color[2]
	mask_img = mask_img.astype(np.uint8)
	img_out = Image.fromarray(mask_img.transpose([1, 2, 0]), 'RGB')
	return img_out


def draw_vector_field(field: np.ndarray) -> np.ndarray:
	"""
	draw the vector field with color
	:param field: the vector field of shape (2, h, w) with order (x, y)
	:return: vector field image array of shape (h, w, 3)
	"""
	h, w = field.shape[1], field.shape[2]
	image = np.zeros((3, h, w))  # rgb
	# compute the angle at each pixel location in degree
	angles = np.arctan2(field[1], field[0]) * (180.0 / np.pi)  # shape (h, w)
	# index the angle values, we discretize the angles into 8 bins
	angles[(angles >= 0) & (angles < 45)] = 0
	angles[(angles >= 45) & (angles < 90)] = 1
	angles[(angles >= 90) & (angles < 135)] = 2
	angles[(angles >= 135) & (angles < 180)] = 3
	angles[(angles >= -180) & (angles < -135)] = 4
	angles[(angles >= -135) & (angles < -90)] = 5
	angles[(angles >= -90) & (angles < -45)] = 6
	angles[(angles >= -45) & (angles < 0)] = 7
	# given the image color
	for i in range(8):
		# image[0, angles == i] = VECTOR_FIELD_COLOR[i][0]
		# image[1, angles == i] = VECTOR_FIELD_COLOR[i][1]
		# image[2, angles == i] = VECTOR_FIELD_COLOR[i][2]
		# the 3 expressions above are equal to the one below
		image[:, angles == i] = np.array(VECTOR_FIELD_COLOR[i]).reshape((-1, 1))

	# cv2.imshow('Map', image.transpose([1, 2, 0]).astype(np.uint8))
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	return image.transpose([1, 2, 0]).astype(np.uint8)  # do not forget to convert to np.unit8 for the image representation


# Module testing
if __name__ == '__main__':
	# import torch
	# test label mask drawing function
	# img_ori = Image.open(r'E:\1Downloaded\datasets\LINEMOD_from_yolo-6d\benchvise\mask\0000.png')
	# img_arr = np.array(img_ori)
	# img_arr[img_arr == 255] = 1
	# mas = np.zeros((13, 480, 640))
	# mas[1, :, :] = img_arr[:, :, 0]
	# linemod_label = draw_linemod_label(mas)
	# linemod_label.show()

	# test vector map drawing
	# from datasets import LinemodDatasetProvider
	# from utils import LinemodOutputExtractor

	# txt = r'E:\1Downloaded\datasets\LINEMOD_from_yolo-6d\benchvise\labels\000000.txt'
	# lab, maps = LinemodDatasetProvider.provide_keypoints_vector_maps(txt)
	# print(lab)
	# print(maps.shape)
	# l, coordinates = LinemodDatasetProvider.provide_keypoints_coordinates(txt)
	# print(coordinates.shape)
	# # draw the coordinate map
	# img_map = draw_vector_field(LinemodOutputExtractor.extract_vector_field_by_name(maps.numpy(), 'benchvise')[4:6])
	# print(img_map.shape)
	# img_vector = Image.fromarray(img_map, 'RGB')
	# img_vector = draw_points(img_vector, coordinates, need_text=True)
	# img_vector.show()

	# test mask one-hot converter
	m = np.array([[0, 1, 2],
	              [0, 0, 0],
	              [2, 1, 2],
	              [1, 2, 0]])
	print(mask_onehot_converter(m, 3))
