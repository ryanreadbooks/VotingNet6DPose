"""
@ Author: ryanreadbooks
@ Time: 9/25/2020, 15:00
@ File name: generate_model_kps.py
@ File description: Generate the keypoints of the model by farthest point sampling algorithm
"""

import os
import numpy as np
from utils.inout import load_model_points
from .farthest_point_sampling import farthest_point_sampling
from utils.geometry_utils import non_homo_to_homo
from datasets import LinemodDatasetProvider


def generate_model_kps(path: str):
	"""
	Generate keypoints bu FPS for one object
	:param path: path of given object dataset. The root of the model data
	"""
	model_pts: np.ndarray = load_model_points(path)  # shape (n, 3)
	kps_indices: np.ndarray = farthest_point_sampling(pc=model_pts, n_points=8)  # get the indices of the selected points
	model_kps: np.ndarray = model_pts[kps_indices, :]  # shape (8, 3)
	# project the model keypoints onto all images of model
	model_kps_homo: np.ndarray = non_homo_to_homo(model_kps)
	# load all poses and ready to project
	poses_path = os.path.walk()
	rotation: np.ndarray = LinemodDatasetProvider.provide_rotation(rot_path)
	translation: np.ndarray = LinemodDatasetProvider.provide_rotation(tra_path)
	pose = np.hstack([rotation, translation])
	# todo implement later
