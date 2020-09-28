"""
@ Author: ryanreadbooks
@ Time: 9/7/2020, 19:17
@ File name: evaluators.py
@ File description: the evaluation of the model is implemented in this file
"""
import os
import time
import statistics
from typing import List, Dict, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as Data
from PIL import Image

from configs import constants
from datasets import LinemodDatasetProvider, Linemod
from evaluator import VoteProcedure
import evaluator.metrics as metrics
from utils import LinemodOutputExtractor, draw_3d_bbox
from utils import geometry_utils
from utils import ICPRefinement


class LinemodEvaluator(object):
	def __init__(self, network: torch.nn.Module, category: str, refinement: bool = False, simple: bool = False,
	             need_model_origin: bool = False):
		"""
		Init function
		:param network: the network model to be evaluated
		:param category: which class the model belongs
		:param refinement: use icp refinement of not, default=False
		:param simple: whether to use simple format or not, when in simple situation, no OutputExtractor is needed
		"""
		super(LinemodEvaluator, self).__init__()
		if category not in constants.LINEMOD_OBJECTS_NAME:
			raise ValueError('category {} is not in the list of LINEMOD dataset'.format(category))
		model_path = os.path.join(constants.DATASET_PATH_ROOT, category, category + '.xyz')
		self.category = category
		self.model_points, self.model_keypoints = LinemodDatasetProvider.provide_model_points(model_path)
		self.need_model_origin = need_model_origin
		if need_model_origin:
			temp = np.zeros((self.model_keypoints.shape[0] + 1, 3), dtype=np.float32)
			temp[1:, :] = self.model_keypoints
			print(temp)
			self.model_keypoints = temp
		self.diameter = constants.LINEMOD_OBJ_DIAMETER[category]  # unit: cm
		assert self.diameter != 0, "Diameter of object {} is 0, plz fill in".format(self.category)
		self.network = network
		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.network.to(self.device)
		self.network.eval()

		self.voting_procedure = VoteProcedure((constants.LINEMOD_IMG_HEIGHT, constants.LINEMOD_IMG_WIDTH))
		self.refinement = None
		self.simple = simple
		if refinement:
			self.refinement = ICPRefinement(category=category)

	def predict_keypoints(self, pred_mask: np.ndarray, pred_vector_map: np.ndarray) -> np.ndarray:
		"""
		process pred_mask and pred_vector_map and get keypoints from them
		:param pred_mask: predicted mask of an object, shape of (h, w), which means that the output of network needs processing
		:param pred_vector_map: predicted vector map of shape (2 * num_keypoints, h ,w)
		:return: the predicted keypoints, array of shape (num_of_keypoints, 2), each row is of the format of (x, y)
		"""
		pred_keypoints: np.ndarray = self.voting_procedure.provide_keypoints(mask=pred_mask, vmap=pred_vector_map)
		return pred_keypoints

	def evaluate(self, add_threshold: float = 0.1, proj_threshold: int = 5, angle_threshold: float = 5.,
	             trans_threshold: float = 5.) -> Dict:
		"""
		Evaluate the whole network model. Evaluate one batch of testing image. Calculate several metrics
		:param add_threshold: the ADD(-S) metrics threshold. ADD(-s) error smaller than 'threshold * diameter' is considered correct, default=0.1
		:param proj_threshold: the projection error [px] threshold. Smaller than threshold is considered correct
		:param angle_threshold: rotation error threshold in in 5cm5°
		:param trans_threshold: translation error threshold in 5cm5°,
		:return: accuracy dict with keys: 'add', 'add-s', '5cm5degree', 'projection', 'miou'. Containing accuracy of all metrics
		"""

		image_transform = transforms.Compose([transforms.ToTensor(),
		                                      transforms.Normalize(mean=constants.IMAGE_MEAN, std=constants.IMAGE_STD)])
		linemod_dataset = Linemod(root_dir=constants.DATASET_PATH_ROOT,
		                          train=False, category=self.category,
		                          dataset_size=50,
		                          transform=image_transform,
		                          need_bg=True,
		                          onehot=False,
		                          simple=self.simple)
		dataloader = Data.DataLoader(linemod_dataset, batch_size=2, pin_memory=True)

		# init metrics storing space
		miou_list = list()
		add_acc_list = list()
		adds_acc_list = list()
		rot_tra_arr_list = list()
		projection_acc_list = list()

		add_acc_list_refined = list()
		adds_acc_list_refined = list()
		rot_tra_arr_list_refined = list()
		projection_acc_list_refined = list()

		for i, data in enumerate(dataloader):
			mask_label: torch.Tensor
			img, mask_label, vmap_label, label, img_path = data  # all torch.Tensor
			img, label = img.to(self.device), label.to(self.device)
			batch_size = img.shape[0]
			out: Tuple = self.network(img)
			pred_masks: torch.Tensor = out[0]  # shape (n, 13+1(or 1+1=2), h, w)
			pred_vmap: np.ndarary = out[1].cpu().detach().numpy()  # shape (n, 234(or 18), h, w)
			# use softmax and argmax to find the probability
			pred_masks: torch.Tensor = torch.softmax(pred_masks, dim=1)
			binary_masks: torch.Tensor = pred_masks.argmax(dim=1)  # shape (n, h, w)
			# convert the pred_mask to binary mask, shape (n, h, w)
			binary_masks: np.ndarray = torch.where(binary_masks == label[0],
			                                       torch.tensor(1).to(self.device),
			                                       torch.tensor(0).to(self.device)).cpu().detach().numpy()
			# convert tensor non-binary mask gt to numpy binary mask gt, shape (n, h, w)
			binary_mask_gt: np.ndarray = torch.where(mask_label.cpu() == label[0].cpu().item(),
			                                         torch.tensor(1.),
			                                         torch.tensor(0.)).cpu().detach().numpy()
			# metrics:
			# mask iou
			miou = metrics.mask_miou(binary_masks, binary_mask_gt)
			miou_list.append(miou)
			# process every image in batch one by one
			for j in range(batch_size):
				if not self.simple:
					obj_vmap: np.ndarray = LinemodOutputExtractor.extract_vector_field_by_name(pred_vmap[j], name=self.category)
				else:
					obj_vmap: np.ndarray = pred_vmap[j]
				# shape (9, 2), the first point is the location of object center
				pred_keypoints: np.nadarry = self.voting_procedure.provide_keypoints(binary_masks[j], obj_vmap)
				# predicted pose, (3, 4)
				pred_pose: np.ndarray = geometry_utils.solve_pnp(object_pts=self.model_keypoints,
				                                                 image_pts=pred_keypoints if self.need_model_origin else pred_keypoints[1:],
				                                                 camera_k=constants.CAMERA)  # shape (3, 4)
				# Load the GT rotation and translation
				gt_pose: np.ndarray = LinemodDatasetProvider.provide_pose(img_path[j])  # shape (3, 4)
				image_label_path = img_path[j].replace('JPEGImages', 'labels')
				image_label_path = image_label_path.replace('jpg', 'txt')
				gt_keypoints = LinemodDatasetProvider.provide_keypoints_coordinates(image_label_path)[1].numpy()
				print('GT Keypoints: \n', gt_keypoints)
				print('Pred Keypoints: \n', pred_keypoints)

				if self.refinement is not None:
					# do the icp process
					depth_arr: np.ndarray = LinemodDatasetProvider.provide_depth(img_path[j])
					pred_pose_refined: np.ndarray = self.refinement.refine(depth=depth_arr, mask=binary_masks[j], pose=pred_pose,
					                                                       model_pts=self.model_points)
					add_error_refined: float = metrics.calculate_add(pred_pose_refined, gt_pose, self.model_points)
					adds_error_refined: float = metrics.calculate_add_s(pred_pose_refined, gt_pose, self.model_points)
					rot_error_refined: float = metrics.rotation_error(pred_pose_refined[:, :3], gt_pose[:, :3])
					tra_error_refined: float = metrics.translation_error(pred_pose_refined[:, -1], gt_pose[:, -1])
					proj_error_refined: float = metrics.projection_error(pts_3d=self.model_points, camera_k=constants.CAMERA,
					                                                     pred_pose=pred_pose_refined,
					                                                     gt_pose=gt_pose)

					# according to the errors, determine the pose is correct or not
					add_acc_list_refined.append(add_error_refined < add_threshold * self.diameter)
					adds_acc_list_refined.append(adds_error_refined < add_threshold * self.diameter)
					rot_tra_arr_list_refined.append(rot_error_refined < angle_threshold and tra_error_refined < trans_threshold)
					projection_acc_list_refined.append(proj_error_refined < proj_threshold)

				# calculate all kinds of errors
				add_error: float = metrics.calculate_add(pred_pose, gt_pose, self.model_points)
				adds_error: float = metrics.calculate_add_s(pred_pose, gt_pose, self.model_points)
				rot_error: float = metrics.rotation_error(pred_pose[:, :3], gt_pose[:, :3])
				tra_error: float = metrics.translation_error(pred_pose[:, -1], gt_pose[:, -1])
				proj_error: float = metrics.projection_error(pts_3d=self.model_points, camera_k=constants.CAMERA,
				                                             pred_pose=pred_pose,
				                                             gt_pose=gt_pose)

				# rewrite the check_pose_correct function
				add_acc_list.append(add_error < add_threshold * self.diameter)
				adds_acc_list.append(adds_error < add_threshold * self.diameter)
				rot_tra_arr_list.append((rot_error < angle_threshold) and (tra_error < trans_threshold))
				projection_acc_list.append(proj_error < proj_threshold)

		# summary all metrics
		accuracies: Dict = self.summary(add_acc_list, adds_acc_list, rot_tra_arr_list, projection_acc_list, miou_list)

		if self.refinement is not None:
			accuracies_refined: Dict = self.summary(add_acc_list_refined, adds_acc_list_refined, rot_tra_arr_list_refined,
			                                        projection_acc_list_refined,
			                                        miou_list)
			print('Accuracy with refinement: \n', accuracies_refined)

		# add threshold info
		accuracies['add_thres'] = add_threshold
		accuracies['proj_thres'] = proj_threshold
		accuracies['rot_thres'] = angle_threshold
		accuracies['tra_thres'] = trans_threshold

		# print('Accuracy: \n', accuracies)

		return accuracies

	def summary(self, add: List, add_s: List, rot_tra: List, projection: List, miou: List) -> Dict:
		"""
		Summarize all the metrics, calculate the average accuracy.
		:param add: ADD metrics accuracy list
		:param add_s: ADD-s metrics accuracy list
		:param rot_tra: 5cm5° metrics accuracy list
		:param projection: Projection metrics accuracy list
		:param miou: the mean-IoU of mask
		:return: a dict that has all metrics accuracy
		"""
		result = dict()
		result['category'] = self.category
		result['add'] = np.mean(add)
		result['add-s'] = np.mean(add_s)
		result['5cm5degree'] = np.mean(rot_tra)
		result['projection'] = np.mean(projection)
		result['miou'] = np.mean(miou)
		return result

	def pipeline(self, img_path: str) -> np.ndarray:
		"""
		The pose estimation pipeline. This function takes a image path as input, output the prediction image
		:param img_path: the path of the detected image
		:return: the predicted pose
		"""

		test_image = Image.open(img_path)
		image_transform = transforms.Compose([transforms.ToTensor(),
		                                      transforms.Normalize(mean=constants.IMAGE_MEAN, std=constants.IMAGE_STD)])
		img_arr: torch.Tensor = image_transform(test_image)[None].to(self.device)  # shape (1, 3, h, w)
		print('=======================================')
		print('Model loaded into {}, evaluation starts...'.format(self.device))

		inference_start_time = time.time()
		net_out = self.network(img_arr)
		inference_took_time = time.time() - inference_start_time
		print('Model inference took time: {:.6f}'.format(inference_took_time))

		pred_mask: torch.Tensor = net_out[0]
		pred_vector_map: np.ndarray = net_out[1].cpu().detach().numpy()
		# probability of the category at every pixel location
		pred_mask = torch.softmax(pred_mask, dim=1)
		# make it to binary mask, 0-background, 255-object
		binary_mask = pred_mask.argmax(dim=1, keepdim=True)[0, 0]

		if constants.EVALUATOR_RESULTS_PATH == '':
			# default path for saving the results
			mask_save_path = 'log_info/results/predicted_{}_mask.png'.format(self.category)
		else:
			mask_save_path = os.path.join(constants.EVALUATOR_RESULTS_PATH, 'predicted_{}_mask.png'.format(self.category))

		# voting procedure
		# extract the correspondence from vector map，shape (18, h, w)
		if not self.simple:
			object_vector_map: np.ndarray = LinemodOutputExtractor.extract_vector_field_by_name(pred_vector_map[0], self.category)
			cls_label: int = constants.LINEMOD_OBJECTS_NAME.index(self.category)
		else:
			object_vector_map: np.ndarray = pred_vector_map[0]
			cls_label: int = 0
		# visualize the binary mask
		binary_mask = torch.where(binary_mask == cls_label, torch.tensor(255).to(self.device), torch.tensor(0).to(self.device))
		binary_mask_np: np.ndarray = binary_mask.cpu().detach().numpy().astype(np.uint8)
		# save the mask result
		Image.fromarray(binary_mask_np, 'L').save(mask_save_path)

		# from image mask to 0-1 mask
		object_binary_mask = np.where(binary_mask_np == 255, 1, 0)  # shape (h, w)
		# we can get the keypoints now
		voting_start_time = time.time()
		pred_keypoints: np.ndarray = self.voting_procedure.provide_keypoints(object_binary_mask, object_vector_map)

		voting_took_time = time.time() - voting_start_time

		pred_pose: np.ndarray = geometry_utils.solve_pnp(object_pts=self.model_keypoints,
		                                                 image_pts=pred_keypoints[1:],
		                                                 camera_k=constants.CAMERA)  # shape (3, 4)
		print('The voting procedure took time: {:.6f}'.format(voting_took_time))
		print('Predicted Keypoints are listed below:\n', pred_keypoints)

		test_image_with_box = draw_3d_bbox(test_image, pred_keypoints[1:], 'blue')

		# save the result
		box_save_path = constants.EVALUATOR_RESULTS_PATH + '/predicted_{}_3dbox.png'.format(self.category)
		test_image_with_box.save(box_save_path)
		total_time = inference_took_time + voting_took_time
		print('Pipeline took total time: {:.6f}'.format(total_time))
		print('=======================================')

		return pred_pose
