"""
@ Author: ryanreadbooks
@ Time: 9/7/2020, 19:20
@ File name: trainer.py
@ File description: Provide the Trainer to train the network
"""

import os
import time

from tqdm import tqdm
import torch
from torch.backends import cudnn
import torch.utils.data as torch_data
from tensorboardX import SummaryWriter

# from configs import training_configs as cfgs, constants
from configs.configuration import train_config as cfgs, regular_config


class Trainer(object):
    def __init__(self, network: torch.nn.Module, mask_loss_fn: torch.nn.Module, vector_loss_fn: torch.nn.Module):
        """
        init function
        :param network: the model you want to train
        :param mask_loss_fn: the loss function of mask branch
        :param vector_loss_fn: the loss function of vector map branch
        """
        self.network = network
        self.lr = cfgs.lr
        self.batch_size = cfgs.batch_size
        self.weight_decay = cfgs.weight_decay
        self.momentum = cfgs.momentum
        self.epochs = cfgs.epochs
        self.milestone = cfgs.milestones
        self.log_train = cfgs.log_train
        self.scheduler = None
        self.use_cuda = cfgs.cuda
        self.params = [p for p in network.parameters() if p.requires_grad]

        if self.log_train:
            writer_dir = os.path.join(regular_config.project_dir, 'log_info', 'train_log')
            self.log_writer = SummaryWriter(writer_dir)

        # optimizer definition
        optim_option: str = cfgs.optimizer
        if optim_option == 'sgd':
            self.optimizer = torch.optim.SGD(self.params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif optim_option == 'adam':
            self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.weight_decay)
        # scheduler definition
        scheduler_type = cfgs.scheduler
        if scheduler_type != '':
            gamma = cfgs.gamma
            if scheduler_type == 'step':
                step = cfgs.frequency
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step, gamma=gamma)
            elif scheduler_type == 'multistep':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestone, gamma=gamma)
            elif scheduler_type == 'exponential':
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
        cudnn.benchmark = True
        cudnn.deterministic = True
        if self.use_cuda:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            if device == torch.device('cpu'):
                print('Configure to use cuda, but not gpu detected, already use cpu instead')
        else:
            device = torch.device('cpu')
        print('Model loaded into {}...\nTraining starts...'.format(device))
        self.network.to(device)
        self.mask_loss_fn.to(device)
        self.vector_map_loss_fn.to(device)
        self.network.train(True)

        vector_map_channel = regular_config.num_keypoint * 2
        if resume:
            # resume training from the last state
            if path is None:
                raise ValueError('when you want to resume training, you need to specify a path to load the last state')
            last_state = torch.load(path)
            # todo: should handle more complex situation here, such as epoch, optimizer, learning rate and so on...
            self.network.load_state_dict(last_state)

            # check if the model fits the configuration
            if self.network.vector_branch.conv11.out_channels != vector_map_channel:
                raise ValueError('The output channel of the vector branch of the loaded model does not match your num-keypoint in config.xml')
            print('resuming training...')

        # start training
        for epoch in range(self.epochs):
            total_loss = torch.tensor(0.0).to(device)
            with tqdm(desc='[TRAIN_LOG_INFO - Epoch {:d}/{:d}]'.format(epoch + 1, self.epochs), total=len(dataloader), unit='batch', dynamic_ncols=True) as pbar:
                for i, batch_data in enumerate(dataloader):
                    # training data shape clarification:
                    # color_img: shape (n, 3, h, w)
                    # mask_target: shape (n, h, w)(no one-hot) or (n, num_class + 1, h, w)(one-hot)
                    # vector_map_target: shape (n, n_class * n_keypoints * 2, h, w)
                    # cls_target: class label, shape (1, n)
                    color_img, mask_target, vector_map_target, cls_target, _ = batch_data
                    color_img, mask_target, vector_map_target, = color_img.to(device), mask_target.to(device), vector_map_target.to(device)
                    self.optimizer.zero_grad()
                    # forward pass
                    # mask_pred: shape (n, num_class + 1, h, w)
                    # vector_map_pred: shape (n, n_class * n_keypoints * 2, h, w)
                    mask_pred, vector_map_pred = self.network(color_img)
                    # calculate loss of this one batch with many data, we use multi-loss here
                    mask_loss = self.mask_loss_fn(mask_pred, mask_target[:, None, :, :])

                    # we only calculate the vector loss of the object pixel
                    weights = mask_target[:, None, :, :].expand(-1, vector_map_channel, -1, -1)
                    weights = weights[:, None, :, :]  # shape (n, 1, h, w), I have to reshape it here
                    weights = weights.type(torch.float32).to(device) * 5
                    vector_loss = self.vector_map_loss_fn(vector_map_pred * weights, vector_map_target * weights)
                    vector_loss = vector_loss / weights.sum() / mask_target.shape[0]
                    total_loss = 2 * mask_loss + 2 * vector_loss
                    total_loss.backward(retain_graph=True)
                    # choose to clip the gradients for stability
                    torch.nn.utils.clip_grad_value_(self.params, 50.)

                    self.optimizer.step()
                    # update the pbar after one batch is processed
                    running_lr = self.lr
                    if self.scheduler is not None:
                        running_lr = self.scheduler.get_last_lr()

                    postfix = {'Mask loss': mask_loss.item(), 'Vector loss': vector_loss.item(), 'Total loss': total_loss.item(),
                               'Lr': running_lr}
                    pbar.set_postfix(postfix)
                    pbar.update()
                    if self.log_train:
                        self.log_writer.add_scalar('MaskLoss', mask_loss.item(), epoch)
                        self.log_writer.add_scalar('VectorLoss', vector_loss.item(), epoch)
                        self.log_writer.add_scalar('TotalLoss', total_loss.item(), epoch)
                        self.log_writer.add_scalar('LearningRate', running_lr, epoch)

            # one epoch ended, should use scheduler.step()
            if self.scheduler is not None:
                print('Lr drop..')
                self.scheduler.step()

            # hit milestone, save it
            if (epoch + 1) in self.milestone:
                saved_state_dict = self.network.state_dict()
                dataset_name = regular_config.dataset_name
                category_name = regular_config.category
                kps_type = regular_config.keypoint_type
                mode = regular_config.mode
                saved_path = regular_config.model_saved_path
                saved_filename = '{:s}_{:s}_{:s}_{:s}_epoch{:d}_loss{:.6f}.pth'.format(dataset_name, category_name, kps_type, mode, epoch + 1, total_loss.item())
                if saved_path == '':
                    # Default path for saving the model params
                    saved_path = os.path.join(regular_config.project_dir, 'log_info', 'models')
                save_path = os.path.join(saved_path, saved_filename)
                print('Hit milestone at epoch {:d}, model has been saved at {:s}'.format(epoch + 1, save_path))
                torch.save(saved_state_dict, save_path)


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
