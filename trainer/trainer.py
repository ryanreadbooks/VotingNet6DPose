"""
@ Author: ryanreadbooks
@ Time: 9/7/2020, 19:20
@ File name: trainer.py
@ File description: Provide the Trainer to train the network
"""

import os
import time
import torch
import torch.utils.data as torch_data
from configs import training_configs as cfgs, constants


class Trainer(object):
	def __init__(self, network: torch.nn.Module, mask_loss_fn: torch.nn.Module, vector_loss_fn: torch.nn.Module):
		"""
		init function
		:param network: the model you want to train
		:param mask_loss_fn: the loss function of mask branch
		:param vector_loss_fn: the loss function of vector map branch
		"""
		train_cfgs = cfgs.TRAINING_CONFIGS
		self.network = network
		# extract the params that needs training
		self.lr = train_cfgs['lr']
		self.batch_size = train_cfgs['batch_size']
		self.weight_decay = train_cfgs['weight_decay']
		self.momentum = train_cfgs['momentum']
		self.epoch = train_cfgs['epoch']
		self.milestone = train_cfgs['milestone']
		self.scheduler = None
		self.params = [p for p in network.parameters() if p.requires_grad]

		# optimizer definition
		optim_option: str = train_cfgs['optimizer'].lower()
		if optim_option == 'sgd':
			self.optimizer = torch.optim.SGD(self.params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
		elif optim_option == 'adam':
			self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.weight_decay)
		else:
			raise ValueError('optimizer {} is not supported'.format(optim_option))
		# scheduler definition
		if train_cfgs['scheduler']:
			scheduler_type = train_cfgs['scheduler_type']
			gamma = train_cfgs['lr_drop_gamma']
			if scheduler_type == 'step_lr':
				step = train_cfgs['lr_drop_per_epoch']
				self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step, gamma=gamma)
			elif scheduler_type == 'multistep_lr':
				self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestone, gamma=gamma)
			elif scheduler_type == 'exp_lr':
				self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

		# loss function definition
		self.vector_map_loss_fn = vector_loss_fn
		self.mask_loss_fn = mask_loss_fn

	def train(self, dataloader: torch_data.DataLoader, resume: bool = False, path: str = None):
		"""
		start the training process
		:param dataloader: the dataloader containing the training data
		:param resume: continue training based on the info given by path
		:param path: the path that gives the last training state
		:return:
		"""
		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		print('Model loaded to {}...\nTraining starts...'.format(device))
		self.network.to(device)
		self.mask_loss_fn.to(device)
		self.vector_map_loss_fn.to(device)
		self.network.train(True)
		if resume:
			# resume training from the last state
			if path is None:
				raise ValueError('when you want to resume training, you need to specify a path to load the last state')
			last_state = torch.load(path)
			# todo: should handle more complex situation here, such as epoch, optimizer, learning rate and so on...
			self.network.load_state_dict(last_state)
			print('resuming training...')

		# start training
		for epoch in range(self.epoch):
			epoch_start_time = time.time()
			multi_loss = torch.tensor(0.0).to(device)
			for i, batch_data in enumerate(dataloader):
				# training data shape clarification:
				# color_img: shape (n, 3, h, w)
				# mask_target: shape (n, h, w)(no onehot) or (n, num_class + 1, h, w)(onehot)
				# vector_map_target: shape (n, n_class * n_keypoints * 2, h, w)
				# cls_target: class label, shape (1, n)
				color_img, mask_target, vector_map_target, cls_target, _ = batch_data
				color_img, mask_target, vector_map_target = color_img.to(device), mask_target.to(device), vector_map_target.to(device)
				self.optimizer.zero_grad()
				# forward pass
				# mask_pred: shape (n, num_class + 1, h, w)
				# vector_map_pred: shape (n, n_class * n_keypoints * 2, h, w)
				mask_pred, vector_map_pred = self.network(color_img)
				# calculate loss of this one batch with many data, we use multi-loss here
				mask_loss = self.mask_loss_fn(mask_pred, mask_target)

				# we only calculate the vector loss of the object pixel
				cls_label_reshaped: torch.Tensor = cls_target.reshape((-1, 1, 1)).to(device)
				# weight same shape as mask_target
				weights = torch.where(mask_target == cls_label_reshaped, torch.tensor(1., device=device), torch.tensor(0., device=device))
				weights = weights[:, None, :, :]  # shape (n, 1, h, w), I have to reshape it here
				weights = weights.type(torch.float32).to(device)
				vector_loss = self.vector_map_loss_fn(vector_map_pred * weights, vector_map_target * weights)
				vector_loss = vector_loss / weights.sum() / mask_target.shape[0]
				multi_loss = mask_loss + vector_loss
				multi_loss.backward(keep_graph=True)
				# choose to clip the gradients for stability
				torch.nn.utils.clip_grad_value_(self.params, 40.)

				if i % 50 == 0:
					print('Checking loss: mask loss: {:.10f}, vector loss: {:.6f}'.format(mask_loss, vector_loss))
					print('Checking loss: {:.6f} in epoch {:d} with step {:d}: '.format(multi_loss, epoch, i))

			# step the optimizer
			self.optimizer.step()
			if self.scheduler is not None:
				self.scheduler.step(epoch)

			epoch_took_time = time.time() - epoch_start_time
			# epoch ended, log it
			running_lr = self.lr
			if self.scheduler is not None:
				running_lr = self.scheduler.get_lr()
			log_info = '{:d} of Epochs {:d} -- Loss: {:.5f} -- Lr: {} -- Took time: {:.5f} seconds' \
				.format(epoch, self.epoch, multi_loss.item(), running_lr, epoch_took_time)
			print(log_info)
			# hit milestone, save it
			if (epoch + 1) in self.milestone:
				timestamp = time.time()
				# save_info = {
				# 	'model': self.network.state_dict(),
				# 	'lr': running_lr,
				# 	'epoch': epoch,
				# 	'timestamp': timestamp
				# }
				# todo: learn more about torch.save and .pth format
				save_info = self.network.state_dict()
				if constants.MODEL_SAVE_PATH == '':
					# Default path for saving the model params
					save_path = 'log_info/' + 'simple_cat_{:d}.pth'.format(epoch)
				else:
					save_path = os.path.join(constants.MODEL_SAVE_PATH, 'simple_cat_{:d}.pth'.format(epoch))
				torch.save(save_info, save_path)


class CrossEntropy(torch.nn.Module):

	def __init__(self) -> None:
		super(CrossEntropy, self).__init__()

	def forward(self, pred, target):
		"""
		segmentation loss
		:param pred: prediction (batch_size, n_class, h, w), it already softmax
		:param target: ground truth label (batch_size, n_class, h, w)
		:return: loss value
		"""
		batch_size, c, h, w = pred.size()
		# we don't need to divide this by the number of channel
		loss = -torch.sum((target * torch.log(pred))) / (batch_size * h * w)
		return loss
