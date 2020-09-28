"""
@ Author: ryanreadbooks
@ Time: 9/7/2020, 19:18
@ File name: geometry_utils.py
@ File description: define a bunch of helper functions that are related to the object model and geometry
"""

import numpy as np
import cv2


def get_model_corners(model_pts: np.ndarray) -> np.ndarray:
	"""
	return the 8 corners of a model
	:param model_pts: model point cloud, shape of (~, 3), ~ means the number of points in model point cloud
	:return: the model corners, array of shape (8, 3)
	"""
	mins = np.min(model_pts, axis=0)
	maxs = np.max(model_pts, axis=0)
	min_x, min_y, min_z = mins[0], mins[1], mins[2]
	max_x, max_y, max_z = maxs[0], maxs[1], maxs[2]
	# vertices = np.array([
	# 	[min_x, min_y, min_z],
	# 	[min_x, min_y, max_z],
	# 	[min_x, max_y, min_z],
	# 	[min_x, max_y, max_z],
	# 	[max_x, min_y, min_z],
	# 	[max_x, min_y, max_z],
	# 	[max_x, max_y, min_z],
	# 	[max_x, max_y, max_z]])
	vertices = np.array([
		[min_x, max_y, max_z],
		[min_x, max_y, min_z],
		[min_x, min_y, max_z],
		[min_x, min_y, min_z],
		[max_x, max_y, max_z],
		[max_x, max_y, min_z],
		[max_x, min_y, max_z],
		[max_x, min_y, min_z]])
	return vertices


def non_homo_to_homo(pts) -> np.ndarray:
	"""
	convert non-homogeneous coordinates to homogeneous coordinates
	:param pts: point coordinates array of shape (~, m), m is usually 2 or 3, representing 2d coordinates and 3d coordinates
	:return: the homogeneous coordinates of the input points
	"""
	m = pts.shape[1]
	pts_homo = np.ones((pts.shape[0], m + 1))
	pts_homo[:, :m] = np.copy(pts)
	return pts_homo


def project_3d_2d(pts_3d: np.ndarray, camera_intrinsic: np.ndarray, transformation: np.ndarray) -> np.ndarray:
	"""
	project 3d points to 2d image plane and return
	:param pts_3d: 3d points to be projected, shape of (n, 3)
	:param camera_intrinsic: camera intrinsics, shape of (3, 3)
	:param transformation: the transformation matrix, shape (3, 4), [R|t]
	:return: array of projected points, shape of (n, 2)
	"""
	# convert the 3d points to homogeneous coordinates
	pts_3d_homo = non_homo_to_homo(pts_3d)  # shape (n, 4)
	projected_homo = (camera_intrinsic @ transformation @ pts_3d_homo.T).T  # (3, 3) x (3, 4) x (4, n) = (3, n) -> Transpose (n, 3)
	# make it homo by dividing the last column
	projected_homo = projected_homo / projected_homo[:, 2].reshape((-1, 1))
	projected = projected_homo[:, :2]
	return projected


def generate_camera_intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
	"""
	form a camera intrinsics matrix
	:param fx: fx
	:param fy: fy
	:param cx: cx
	:param cy: cy
	:return: the camera intrinsics matrix of shape (3, 3)
	"""
	camera = np.eye(3)
	camera[0, 0] = fx
	camera[1, 1] = fy
	camera[0, 2] = cx
	camera[1, 2] = cy
	return camera


def transform_pts(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
	"""
	transform points of model according to the pose
	:param points: model points, array with shape (n, 3)
	:param pose: pose array with shape (3, 4), [R|t]
	:return: the transformed points, array with shape (n, 3)
	"""
	assert (points.shape[1] == 3)
	r = pose[:, :3]  # predicted rotation matrix, shape of (3, 3)
	t = pose[:, -1].reshape((3, 1))  # predicted translation, shape of (3, 1)
	points_transformed = r.dot(points.T) + t

	return points_transformed.T


def calculate_object_diameter(object_pts: np.ndarray) -> float:
	"""
	calculate the diameter of the input object which is represented in array
	:param object_pts: 3d points of the object, array with shape (n, 3)
	:return: the diameter
	"""
	if object_pts.shape[0] > 500:
		raise MemoryError('array may be too large, which will crush the computer...!!!')
	from scipy.spatial.distance import cdist
	# distance: shape (n, n)
	distance: np.ndarray = cdist(object_pts, object_pts, 'euclidean')
	return np.max(distance)


def depth_to_point_cloud(camera_k, depth: np.ndarray) -> np.ndarray:
	"""
	convert the depth image to point cloud
	:param camera_k: the camera intrinsics, array with shape (3,3)
	:param depth: the depth image, array with shape (h, w)
	:return: point cloud, array with shape (n, 3)
	"""
	vs, us = depth.nonzero()
	zs: np.ndarray = depth[vs, us]
	xs = ((us - camera_k[0, 2]) * zs) / float(camera_k[0, 0])
	ys = ((vs - camera_k[1, 2]) * zs) / float(camera_k[1, 1])
	pts = np.array([xs, ys, zs]).T
	return pts


def solve_pnp(object_pts: np.ndarray, image_pts: np.ndarray, camera_k: np.ndarray, method=cv2.SOLVEPNP_ITERATIVE):
	"""
	Solve the PnP problem
	:param object_pts: the points in object coordinate, shape (n, 3)
	:param image_pts: the corresponding points in the image, shape (n, 2)
	:param camera_k: the camera intrinsics matrix, shape (3, 3)
	:param method: the method used to solve PnP problem, default=cv2.SOLVEPNP_EPNP
	:return: the calculated transformation matrix, shape (3, 4)
	"""
	assert object_pts.shape[0] == image_pts.shape[0], 'number of points do not match.'
	# dist_coef = np.zeros(shape=[8, 1], dtype='float64')
	_, r_vec, t_vec = cv2.solvePnP(objectPoints=object_pts.astype(np.float64),
	                               imagePoints=image_pts.astype(np.float64),
	                               cameraMatrix=camera_k.astype(np.float64),
	                               distCoeffs=None,
	                               useExtrinsicGuess=True,
	                               flags=method)
	r_mat, _ = cv2.Rodrigues(r_vec)  # from rotation vector to rotation matrix (3, 3)
	transformation = np.hstack([r_mat, t_vec.reshape((-1, 1))])
	return transformation
