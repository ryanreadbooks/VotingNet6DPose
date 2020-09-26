"""
@ Author: ryanreadbooks
@ Time: 9/25/2020, 14:59
@ File name: farthest_point_sampling.py
@ File description: 最远点采样实现
"""
import numpy as np


def farthest_point_sampling(pc: np.ndarray, n_points: int) -> np.ndarray:
	"""
	FPS implementation
	:param pc: points with shape (n, ~), ~ means any values, n is the number of points
	:param n_points: the number of points to be samples
	:return: index of sampled points in the original points set
	"""
	n = pc.shape[0]
	# 采样点向量，存储的是被采样的点在原矩阵中的列索引位置
	# 称作被选点集
	sampled_index = np.zeros(n_points, dtype=np.int)
	# 第一个采样点通过随机的方法得到
	farthest = np.random.randint(0, n)
	# 存储初始的距离
	distance = np.ones(n) * 1e10
	for i in range(n_points):
		# 找到了最远点，把它放进存放被采样点的集合中
		sampled_index[i] = farthest
		# 计算这个点和所有点的距离
		# 先取出这个参考点的值
		point_ref = pc[farthest]
		# 算出距离，但是不开根号，节省运算
		dist = np.sum((pc - point_ref) ** 2, axis=1)
		mask = dist < distance
		# 只有当mask对应位置==True，才会用dist里面的对应位置的距离更新distance对应位置的距离
		# 因为每次的dist都是剩余点与参考点的最新的距离，而distance中的距离是与被选中的点的点集的距离
		# 这样的话，distance中的值则是每个点到被选点集的距离
		# 更新相应距离，找到剩余点到以有点集的最短距离
		distance[mask] = dist[mask]
		# 选择下一个进入被选点集的点
		farthest = distance.argmax(axis=-1)

	return sampled_index


# Module testing
if __name__ == '__main__':
	# test this function
	import matplotlib.pyplot as plt

	# randomly generating 100 points (x, y)
	points = np.random.rand(100, 2)
	plt.scatter(points[:, 0], points[:, 1], c='blue')
	# use FPS to sample 8 points
	sample_id = farthest_point_sampling(points, 8)
	points_sampled = points[sample_id, :]
	plt.scatter(points_sampled[:, 0], points_sampled[:, 1], c='red')
	plt.show()
