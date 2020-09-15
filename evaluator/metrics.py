"""
@ Author: ryanreadbooks
@ Time: 9/4/2020, 19:51
@ File name: metrics.py
@ File description: some metric functions for calculating the scores
"""

import math
from typing import Tuple

import numpy as np
from utils import project_3d_2d, transform_pts
import scipy.spatial as scipy_spatial


def mask_iou(pred: np.ndarray, truth: np.ndarray) -> float:
	"""
	Calculate the iou between two mask
	:param pred: the predicted mask, shape (h, w), with the value of masked region to be  1, background to be value 0
	:param truth: the ground truth mask, shape (h, w), with the value of masked region to be 1, background to be value 0
	:return: iou between these two masks
	"""
	intersection = np.logical_and(truth, pred)
	union = np.logical_or(truth, pred)
	iou = np.sum(intersection) / np.sum(union)
	return iou


def mask_miou(pred: np.ndarray, truth: np.ndarray) -> float:
	"""
	Calculate the mean IoU between masks
	:param pred: the predicted mask, shape (n, h, w)
	:param truth: the ground truth mask, shape (n, h ,w)
	:return: the mean IoU
	"""
	intersection: np.ndarray = np.logical_and(truth, pred).sum(axis=1).sum(axis=1)
	union: np.ndarray = np.logical_or(truth, pred).sum(axis=1).sum(axis=1)
	ious: np.ndarray = intersection / union
	return ious.mean()


def calculate_add(pred_pose: np.ndarray, gt_pose: np.ndarray, points: np.ndarray) -> float:
	r"""
	Calculate the ADD metric. This metrics is more strict to the predicted pose
	:math:`ADD = \sum[(R * x + T) - (R_p * x + T_p)] / m`

	:param pred_pose: the predicted pose, array with shape (3, 4), [R|t]
	:param gt_pose: the ground truth pose, array with shape (3, 4), [R|t]
	:param points: the model points to be transformed, array with shape (n, 3), n is the number of points
	:return: the ADD value
	"""
	points_pred: np.ndarray = transform_pts(points, pred_pose)
	points_gt: np.ndarray = transform_pts(points, gt_pose)

	return np.linalg.norm(points_gt - points_pred, axis=1).mean()


def calculate_add_s(pred_pose: np.ndarray, gt_pose: np.ndarray, points: np.ndarray) -> float:
	r"""
	Calculate the ADD-S metric. This metirc is less strict than ADD metirc
	:math:`ADD-S = \sum_{x1 \in {M}}{min_{x2 \in{M}} (R * x1 + T)-(R_p * x2 + T_p)} / m`

	:param pred_pose: the predicted pose, array with shape (3, 4), [R|t]
	:param gt_pose: the ground truth pose, array with shape (3, 4), [R|t]
	:param points: the model points to be transformed
	:return: the ADD-S value
	"""

	points_pred: np.ndarray = transform_pts(points, pred_pose)
	points_gt: np.ndarray = transform_pts(points, gt_pose)
	kdtree: scipy_spatial.cKDTree = scipy_spatial.cKDTree(points_pred)
	query_result: Tuple[np.ndarray, np.ndarray] = kdtree.query(points_gt)  # query result: distances, index at points_pred
	nearest_distances = query_result[0]
	add_s: float = nearest_distances.mean()

	return add_s


def rotation_error(r_pred: np.ndarray, r_gt: np.ndarray) -> float:
	"""
	calculate the rotation error between two rotation matrix.
	original implementation from https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/pose_error.py
	math is from Rodrigue's formula.
	:param r_pred: the predicted rotation matrix, shape of (3, 3)
	:param r_gt: the ground truth rotation matrix, shape of (3, 3)
	:return: the calculated error between them
	"""
	assert (r_pred.shape == r_gt.shape == (3, 3)), 'ground truth and predicted value must be of the same shape (3, 3)'
	error_cos = float(0.5 * (np.trace(r_pred.dot(np.linalg.inv(r_gt))) - 1.0))

	# Avoid invalid values due to numerical errors.
	error_cos = min(1.0, max(-1.0, error_cos))

	error = math.acos(error_cos)
	error = 180.0 * error / np.pi  # Convert [rad] to [deg].
	return error


def translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
	"""
	calculate the translation error between two translation vector
	original implementation from https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/pose_error.py
	:param t_pred: predicted translation vector, size of 3
	:param t_gt: ground truth translation vector, size of 3
	:return: the computed translation error
	"""
	assert (t_gt.size == t_pred.size == 3), 'invalid size for translation vector'
	error = np.linalg.norm(t_gt - t_pred)
	return error


def projection_error(pts_3d: np.ndarray, camera_k: np.ndarray, pred_pose: np.ndarray, gt_pose: np.ndarray):
	"""
	Average distance of projections of object model vertices [px]
	:param pts_3d: model points, shape of (n, 3)
	:param camera_k: camera intrinsic matrix, shape of (3, 3)
	:param pred_pose: predicted rotation and translation, shape (3, 4), [R|t]
	:param gt_pose: ground truth rotation and translation, shape (3, 4), [R|t]
	:return: the returned error, unit is pixel
	"""
	# projection shape (n, 2)
	pred_projection: np.ndarray = project_3d_2d(pts_3d=pts_3d, camera_intrinsic=camera_k, transformation=pred_pose)
	gt_projection: np.ndarray = project_3d_2d(pts_3d=pts_3d, camera_intrinsic=camera_k, transformation=gt_pose)
	error = np.linalg.norm(gt_projection - pred_projection, axis=1).mean()
	return error


def check_pose_correct(val1: float, threshold1: float, metric: str, diameter: float = 0., val2: float = 0., threshold2: float = 0.):
	"""
	Check if the pose_pred should be considered correct with respect to pose_gt based on the metric
	:param val1: value to check
	:param threshold1: threshold that determine the values are correct or not
	:param metric: criteria, add, add-s, projection error, 5cm5°
	:param diameter: in add(-s), you need diameter of the object
	:param val2: optional argument for 5cm5° metric
	:param threshold2: optional argument for 5cm5° metric
	:return: pose_pred should be considered correct of not. correct -> True, incorrect -> False
	"""
	if metric == 'add' or metric == 'add-s':
		assert diameter != 0
		if val1 <= threshold1 * diameter:
			return True
		else:
			return False
	elif metric == '5cm5':
		if val1 <= threshold1 and val2 <= threshold2:
			return True
		else:
			return False
	elif metric == 'projection':
		if val1 < threshold1:
			return True
		else:
			return False
