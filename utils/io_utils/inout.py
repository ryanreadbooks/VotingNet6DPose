from typing import Dict
import numpy as np
from PIL import Image
from plyfile import PlyData


def load_model_points(path: str) -> np.ndarray:
	"""
	Load the model points from given path. The given path must be .xyz file of the object, the .ply file does not work here.
	:param path: given path
	:return: the loaded points of the model
	"""
	xyz = []
	with open(path, 'r') as f:
		_ = f.readline()
		while True:
			line = f.readline().rstrip()
			if line == '':
				break
			xyz.append(list(map(float, line.split())))
	xyz = np.float32(xyz)
	return xyz[:, :3]  # we only need the coordinates


def get_ply_model(path):
	"""
	Load the ply model of the object
	"""
	ply = PlyData.read(path)
	data = ply.elements[0].data
	x = data['x']
	y = data['y']
	z = data['z']
	model = np.stack([x, y, z], axis=-1) * 100.  # convert m to cm
	return model


def load_depth_image(path: str) -> np.ndarray:
	# shape (h, w), single channel
	depth: np.ndarray = np.array(Image.open(path)) / 10.  # depth from densefusion(unit is mm), we need cm as unit here
	return depth


def save_dict_to_txt(d: Dict, fp: str) -> None:
	"""
	Save dict to txt file
	:param d: dict to be saved
	:param fp: file name
	:return:
	"""
	content = str()
	for key, value in d.items():
		content += str(key) + ': ' + str(value) + '\n'
	with open(fp, 'w') as f:
		f.write(content)
