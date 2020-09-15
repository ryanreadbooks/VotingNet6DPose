"""
@ Author: ryanreadbooks
@ Time: 9/3/2020 20:09
@ File description: define some decorators to decorate the function, mainly for function annotation purpose
"""
import time
import warnings


def track_running_time(func):
	"""
	used to record function running time
	:param func: function
	:return: wrapper
	"""
	def wrapper(*args, **kwargs):
		start_time = time.time()
		result = func(*args, **kwargs)
		print(f'Finishing {func.__name__} took time {time.time() - start_time}')
		return result
	return wrapper


def deprecated(func):
	"""
	used to annotate the function, indicating that this function is no longer recommended
	:return:
	"""
	def wrapper(*args, **kwargs):
		result = func(*args, **kwargs)
		warnings.warn(message='the {} function may has been deprecated, it is not recommended'.format(func.__name__))
		return result
	return wrapper
