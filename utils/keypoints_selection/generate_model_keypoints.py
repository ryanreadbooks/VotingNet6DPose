"""
@ Author: ryanreadbooks
@ Time: 9/25/2020, 15:00
@ File name: generate_model_keypoints.py
@ File description: Generate the keypoints of the model by farthest point sampling algorithm
"""

import os
from typing import List
import numpy as np
from tqdm import tqdm

from configs.constants import CAMERA
from datasets import LinemodDatasetProvider
from utils.geometry_utils import *
from utils.io_utils import load_model_points
from .farthest_point_sampling import farthest_point_sampling


def generate_model_kps(path: str, model_name: str):
	"""
	Generate keypoints bu FPS for one object
	@param path: path of given object dataset. The root of the model data, like xxx/cat
	@param model_name: model name
	"""
	model_path = os.path.join(path, model_name + '.xyz')
	model_pts: np.ndarray = load_model_points(model_path)  # shape (n, 3)
	kps_indices: np.ndarray = farthest_point_sampling(pc=model_pts, n_points=8)  # get the indices of the selected points
	model_kps: np.ndarray = model_pts[kps_indices, :]  # shape (8, 3)
	# save the model kps
	model_kps_path = os.path.join(path, model_name + '_fps_keypoints.txt')
	# noinspection PyTypeChecker
	np.savetxt(model_kps_path, model_kps)

	# project the model keypoints onto all images of model
	# load all poses and ready to project
	gt_poses_path: str = os.path.join(path, 'gt_poses')
	new_label_saving_base_path = os.path.join(path, 'labels_fps')
	if not os.path.exists(new_label_saving_base_path):
		os.mkdir(new_label_saving_base_path)
	poses_path_list: List = [i for i in os.walk(gt_poses_path)][0][2]
	for i in tqdm(range(len(poses_path_list) // 2)):
		# rotation label and translation label
		rot_path: str = os.path.join(gt_poses_path, 'rot' + str(i) + '.rot')
		tra_path: str = os.path.join(gt_poses_path, 'tra' + str(i) + '.tra')
		rotation: np.ndarray = LinemodDatasetProvider.provide_rotation(rot_path)
		translation: np.ndarray = LinemodDatasetProvider.provide_rotation(tra_path)
		# full pose
		pose: np.ndarray = np.hstack([rotation, translation])
		# project keypoints on to the image
		projected_kps: np.ndarray = project_3d_2d(pts_3d=model_kps, camera_intrinsic=CAMERA, transformation=pose)
		# shape (8, 2), format of (x, y)
		# normalize it
		projected_kps[:, 0] /= 640.
		projected_kps[:, 1] /= 480.
		projected_kps: np.ndarray = projected_kps.reshape((1, -1))
		model_id: int = constants.LINEMOD_OBJECTS_NAME.index(model_name)
		# format the keypoints label
		# format: [model_id, x0, y0, x1, y1, ... , x7, y7, x, x], shape (19, 1)
		new_label: np.ndarray = np.zeros(19)
		new_label[0] = model_id
		new_label[1: -2] = projected_kps
		# write this new_label to the file
		# zfill(): fill the num to size of 6
		new_label_name: str = os.path.join(new_label_saving_base_path, str(i).zfill(6) + '.txt')
		# noinspection PyTypeChecker
		np.savetxt(fname=new_label_name, X=new_label, fmt='%.6f', newline=' ')


def generate_fps_keypoints():
	"""
	generate fps-selected keypoints for all objects
	If you want to call this method, you need to make sure the necessary files exist.
	You need *.xyz object model file, gt_poses folder in which contains ground poses of the model
	"""
	for _, model_name in tqdm(enumerate(constants.LINEMOD_OBJECTS_NAME)):
		path = os.path.join(constants.DATASET_PATH_ROOT, model_name)
		generate_model_kps(path=path, model_name=model_name)
