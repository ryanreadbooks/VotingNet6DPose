"""
@ Author: ryanreadbooks
@ Time: 9/6/2020, 15:10
@ File name: icp.py
@ File description: ICP algorithm (Iterative Closest Point)implementation for post refinement
"""

import numpy as np
import scipy.spatial as scipy_spatial

from utils import geometry_utils
from configs.constants import CAMERA


def fit_two_points(pts_a: np.ndarray, pts_b: np.ndarray):
	"""
	Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions, A->B
	The implementation is the implementation in the website below:
	https://github.com/zju3dv/clean-pvnet/blob/f54fefa1e88cc50cce603f37815a011f3137d137/lib/utils/icp/icp_utils.py#L103
	:param pts_a: points a, NxM numpy array, N is the number of points, M is the dimensions of point
	:param pts_b: points b, NxM numpy array, N is the number of points, M is the dimensions of point
	:return: homogeneous transformation T(m+1, m+1); rotation matrix R(m, m), translation vector t(m, 1)
	"""

	assert pts_a.shape == pts_b.shape, 'shape of points a and b do not match, check again.'
	n, m = pts_a.shape

	# the following codes are the process of one iteration in icp
	centroid_a = np.mean(pts_a, axis=0)
	centroid_b = np.mean(pts_b, axis=0)
	pts_aa = pts_a - centroid_a
	pts_bb = pts_b - centroid_b

	H = np.dot(pts_a.T, pts_bb)
	U, S, V_t = np.linalg.svd(H)
	R = np.dot(V_t.T, U.T)
	# special reflection case
	if np.linalg.det(R) < 0:
		V_t[m - 1, :] *= -1
		R = np.dot(V_t.T, U.T)
	t = centroid_b - np.dot(R, centroid_a.T)

	# homogeneous transformation matrix that maps A to B
	T = np.identity(m + 1)
	T[:m, :m] = R
	T[:m, m] = t

	return T, R, t


def icp_process(pts_a: np.ndarray, pts_b: np.ndarray, init_pose: np.ndarray = None,
                max_iter: int = 200, tolerance: float = 0.001):
	"""
	iteratively use fit_two_points function, that maps a to b
	:param init_pose: initial pose, a homogeneous transformation, shape (m+1, m+1)
	:param pts_a: points a, NxM numpy array, N is the number of points, M is the dimensions of point
	:param pts_b: points b, NxM numpy array, N is the number of points, M is the dimensions of point
	:param max_iter: max iterations, default=200
	:param tolerance: tolerance to end the icp process
	:return: the homogeneous transformation that maps A to B, shape(m+1, m+1)
	"""
	assert pts_a.shape == pts_b.shape, 'shape of points a and b do not match, check again.'
	n, m = pts_a.shape

	# convert them to homogeneous, shape (m+1, n).
	# We take the transpose for matrix multiplication with T
	src_pts: np.ndarray = geometry_utils.non_homo_to_homo(pts_a).T
	dst_pts: np.ndarray = geometry_utils.non_homo_to_homo(pts_b).T

	if init_pose is not None:
		src_pts = np.dot(init_pose, src_pts)  # (m+1, m+1) * (m+1, n) = (m+1, n)

	# init error, this is used as the previous error in the first iteration
	previous_error = 0.

	kdtree = scipy_spatial.cKDTree(dst_pts.T)
	for i in range(max_iter):
		# find the nearest neighbors between the current source and destination points
		distances, indices = kdtree.query(src_pts.T)

		# compute the transformation between the current source and nearest destination points
		T, _, _ = fit_two_points(src_pts[:m, :].T, dst_pts[:m, indices].T)

		# update the src_pts for the next iteration
		# T: shape(m+1, m+1). src_pts: shape(m+1, n)
		src_pts = np.dot(T, src_pts)

		# calculate the current error
		mean_error = np.mean(distances)
		# check the tolerance
		if np.abs(previous_error - mean_error) < tolerance:
			break
		# exit icp process requirement not met
		# update the previous for the next iteration
		previous_error = mean_error

	# iteration process has ended, calculate the final transformation
	T, _, _ = fit_two_points(pts_a, src_pts[:m, :].T)

	return T


class ICPRefinement(object):
	def __init__(self, category: str):
		self.category = category

	def refine(self, depth: np.ndarray, mask: np.ndarray, pose: np.ndarray, model_pts: np.ndarray, camera_k: np.ndarray = CAMERA) -> np.ndarray:
		"""
		refine the estimated pose with icp
		:param mask: the predicted binary mask, shape (h, w)
		:param depth: the depth image(array) which is used for generating the points in real scene, shape (h, w)
		:param pose: the predicted transformation matrix, shape (3, 4), [R|t]
		:param model_pts: points of the model, array with shape (n, 3)
		:param camera_k: the camera intrinsics matrix, shape (3, 3)
		:return: refined pose, shape (3, 4), [R|t]
		"""

		# todo, it seems that it is not correct
		masked_depth = depth * mask
		scene_cloud: np.ndarray = geometry_utils.depth_to_point_cloud(camera_k, depth=masked_depth)  # shape (n, 3)
		m = scene_cloud.shape[0]
		# sample the model points to meet the number of points in scene cloud
		idx: np.ndarray = np.random.choice(model_pts.shape[0], m)
		model_cloud: np.ndarray = model_pts[idx]
		model_cloud: np.ndarray = geometry_utils.transform_pts(model_cloud, pose)
		# start the icp process
		transition_transformation: np.ndarray = icp_process(model_cloud, scene_cloud, tolerance=1e-5)

		# compute the result pose, combining the initial pose
		init_pose = np.eye(4)
		init_pose[:3, :] = pose

		# final pose with refinement
		final_pose: np.ndarray = np.dot(transition_transformation, init_pose)

		return final_pose[:3, :]
