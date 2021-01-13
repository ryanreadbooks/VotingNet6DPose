from typing import Tuple, List
import random
import os

import numpy as np
import torch
import torch.utils.data as torch_utils_data
from PIL import Image
import cv2

from configs import constants
from utils.geometry_utils import get_model_corners
from utils.io_utils.inout import load_model_points
from utils.io_utils import load_depth_image
from configs.configuration import regular_config


class Linemod(torch_utils_data.Dataset):
    """
    LINEMOD dataset in my format, including mask for object in a image, the unit vector field for the image.
    The root path of the dataset folder in your local machine should be organized in the following structure:
        root:
            ape:
                JPEGImages: all the images of this object are in this folder
                labels: all labels of the object in images are here
                mask: all masks of the object in images are here
                gt_poses: all the ground truth poses are in this folder
                depth: all depth images are in this folder
                train.txt: the images(masks/labels) that are used for training
                test.txt: the images(masks/labels) that are used for testing
                ape.ply: the object model
                ape.xyz: the object point cloud
                ...
            benchvise:
                ...
            cam:
                ...
            can:
                ...
            ...
    Attention: make sure the listed sub-folders and files are in one single object folder
    """

    # todo: consider removing the dataset_size argument and use all training data
    def __init__(self, train=True, transform=None):
        """
        init function
        :param train: return data or not, default=True
        :param transform: the transform you want to apply on the image
        """
        super(Linemod, self).__init__()
        self.root_dir = regular_config.data_path_root
        # filenames of all data are stored in this List
        self.dir_list = list()
        self.data_img_path = list()  # list that stores JPEGImages's path
        self.data_mask_path = list()  # list that stores mask image's path
        self.data_labels_path = list()  # list that stores labels's path
        self.transform = transform

        train_or_test = 'train.txt'
        if not train:
            train_or_test = 'test.txt'
        category = regular_config.category
        cls_data_path = os.path.join(self.root_dir, category, train_or_test)
        with open(cls_data_path, 'r') as f:
            lines = f.readlines()
            self.dir_list: List = [line.rstrip()[-10: -4] for line in lines]

        base_dir = os.path.join(self.root_dir, category)
        labels = 'labels'
        if regular_config.keypoint_type == 'fps':
            labels = 'labels_fps'
        for filname in self.dir_list:
            data_img_path = os.path.join(base_dir, 'JPEGImages', filname + '.jpg')
            data_mask_path = os.path.join(base_dir, 'mask', filname[-4:] + '.png')
            data_label_path = os.path.join(base_dir, labels, filname + '.txt')
            self.data_img_path.append(data_img_path)
            self.data_mask_path.append(data_mask_path)
            self.data_labels_path.append(data_label_path)
        train_log = 'training' if train else 'testing'
        print('Found {:d} {:s} data for category {:s}'.format(len(self.dir_list), train_log, category))

    def __getitem__(self, index: int) -> Tuple:  # color, mask, vector maps, cls_label, color_image_path
        """
        Get one data from the dataset
        :param index: index of the dataset
        :return: image itself, mask tensor, coordinate vector, cls_label, color_image_path
        """
        data_img_path: str = self.data_img_path[index]
        data_mask_path: str = self.data_mask_path[index]
        data_label_path: str = self.data_labels_path[index]
        # retrieve relevant information
        cls_label, vector_maps = LinemodDatasetProvider.provide_keypoints_vector_maps(data_label_path)
        color: Image = LinemodDatasetProvider.provide_image(data_img_path)

        # shape (h, w) -> binary mask
        mask: torch.Tensor = LinemodDatasetProvider.provide_mask(data_mask_path)

        # add random transformation

        # transform the image if needed
        if self.transform is not None:
            color = self.transform(color)
        return color, mask, vector_maps, cls_label, data_img_path

    def __len__(self) -> int:
        return len(self.dir_list)


class LinemodDatasetProvider(object):
    """
    Provide the corresponding data label (mask, coordinates of keypoints) according to the given path
    """

    @staticmethod
    def provide_image(path: str) -> Image:
        """
        provide the image according to the given path
        :param path: given path
        :return: returned PIL Image
        """
        return Image.open(path)

    @staticmethod
    def provide_mask(path: str) -> torch.Tensor:
        """
        Provide the simple mask for the network. It is the binary that is returned.
        :param path: given path
        :return: returned binary mask
        """
        mask_single = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # shape (h, w)
        mask = np.where(mask_single == 255, 1, 0)  # we only use 1 and 0 for the mask, 0 for background, 1 for the object

        return torch.from_numpy(mask).float()

    @staticmethod
    def provide_keypoints_coordinates(path: str) -> Tuple[int, torch.Tensor]:
        """
        Provide the keypoints coordinates
        :param path: given path
        :return: cls label, from 0~12;
                 keypoints coordinates of format (x, y), shape (num_of_keypoints, 2)
        """
        with open(path, 'r') as f:
            line = f.readline().strip()
            labels = [i for i in map(lambda x: float(x), line.split(' '))]
            cls_label, kps = labels[0], labels[1:]  # contains class label and keypoints
            keypoints: torch.Tensor = torch.tensor(kps)
            # recover the raw coordinates
            keypoints[::2] *= regular_config.img_width  # x coordinates
            keypoints[1::2] *= regular_config.img_height  # y coordinates
            keypoints: torch.Tensor = torch.stack([keypoints[::2], keypoints[1::2]]).T  # of shape (10, 2)
        return int(cls_label), keypoints[:-1]  # drop the last one that is not keypoint coordinate

    @staticmethod
    def provide_keypoints_vector_maps(path: str) -> Tuple[int, torch.Tensor]:
        """
        Provide the keypoints maps, which is a tensor of shape (num_keypoint x 2 x num_classes, H, W)
        :param path: given path to read the keypoints coordinates and generate keypoints maps,
                    which are unit vectors pointing from every pixel to the keypoint coordinates
        :param simple: whether to return the simple vector map or not. The simple version of it has less output channels.
        :return: class label of the corresponding keypoints;
                corresponding keypoints maps of shape (2 x n_keypoints x n_classes, H, W)
        """
        # generating mesh for all pixels
        width: int = regular_config.img_width
        height: int = regular_config.img_height
        num_of_keypoints: int = regular_config.num_keypoint

        x = torch.linspace(0, width - 1, width)
        y = torch.linspace(0, height - 1, height)
        mesh = torch.meshgrid([y, x])
        # shape (2 * n_keypoints, H, W) with the first channel is x0-coordinate, the second channel is y0-coordinate
        pixel_coordinate = torch.cat([mesh[1][None], mesh[0][None]], dim=0).repeat(num_of_keypoints, 1, 1)
        # load the cls_label and keypoints
        cls_label, keypoints = LinemodDatasetProvider.provide_keypoints_coordinates(path=path)
        channel = 2 * num_of_keypoints  # 2 x n_keypoints

        cls_label = 0  # in simple version, we don't need the cls_label to be specific
        # for unity of the simple and full version of target, we still prepare a space for the target
        keypoints = keypoints.reshape((num_of_keypoints * 2, 1, 1))
        dif = keypoints - pixel_coordinate
        # calculate the norm of dif vector at every pixel location
        # v = (k - p) / |k - p|
        dif_norm: torch.Tensor = dif.reshape((-1, 2, height, width))
        # print(dif_norm.shape)   # shape (n_kp, 2, h, w)
        dif_norm = torch.norm(dif_norm, dim=1)
        # print(dif_norm.shape)   # shape (n_kp, h, w)
        dif_norm: torch.Tensor = dif_norm.reshape((-1, 1, height, width))
        dif_norm: torch.Tensor = dif_norm.repeat(1, 1, 2, 1).reshape((-1, height, width))
        vector_map: torch.Tensor = dif / dif_norm  # shape (2 * n_keypoints, H, W)

        return cls_label, vector_map

    @staticmethod
    def provide_rotation(path: str) -> np.ndarray:
        """
        Load the rotation matrix from Linemod
        :param path: given path
        :return: loaded rotation matrix, shape (3, 3)
        """
        with open(path, 'r') as f:
            lines = f.readlines()[1:]
            tmp_list = list(map(lambda x: x.strip().split(' '), lines))
            rot: np.ndarray = np.array(tmp_list, dtype=np.float32)
        return rot

    @staticmethod
    def provide_translation(path: str) -> np.ndarray:
        """
        Load the translation vector from Linemod
        :param path: given path
        :return: loaded translation vector ,shape (3, 1)
        """
        # it's the same
        return LinemodDatasetProvider.provide_rotation(path)

    @staticmethod
    def provide_pose(path: str) -> np.ndarray:
        """
        Load the transformation matrix from Linemod, given the path
        Input path is just like: 'base_dir/category/JPEGImages/000185.jpg'
        :param path: given path
        :return: transformation matrix, shape (3, 4)
        """
        base_path, file_name = os.path.split(path)
        base_path = base_path.replace('JPEGImages', 'gt_poses')
        file_num = int(os.path.splitext(file_name)[0])
        rot_path = os.path.join(base_path, 'rot' + str(file_num) + '.rot')
        tra_path = os.path.join(base_path, 'tra' + str(file_num) + '.tra')

        rotation: np.ndarray = LinemodDatasetProvider.provide_rotation(rot_path)
        translation: np.ndarray = LinemodDatasetProvider.provide_rotation(tra_path)
        return np.hstack([rotation, translation])

    @staticmethod
    def provide_depth(path: str) -> np.ndarray:
        """
        Load the depth image from given path of Linemod dataset,
        Input path is just like: 'base_dir/category/JPEGImages/000185.jpg'
        :param path: given path
        :return: depth image array, shape with (h, w)
        """
        base_path, file_name = os.path.split(path)
        base_path = base_path.replace('JPEGImages', 'depth')
        file_num = os.path.splitext(file_name)[0][2:]
        depth_path = os.path.join(base_path, file_num + '.png')
        return load_depth_image(depth_path)

    @staticmethod
    def provide_3d_keypoints(path: str):
        """
        Get the 3d keypoints of a model
        :param path: given path
        :return: 3d keypoints
        """
        points = load_model_points(path=path)
        kps: np.ndarray = get_model_corners(points)  # shape (8,3)
        return kps

    @staticmethod
    def provide_model_points(path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the model points and model 3d keypoints
        :param path: given path
        :return: model points; model keypoints
        """
        # todo check correctness
        points: np.ndarray = load_model_points(path=path).astype(np.float64)
        # points: np.ndarray = get_ply_model(path=path)
        kps: np.ndarray = get_model_corners(points).astype(np.float64)  # shape (8,3)
        return points, kps


# Module testing
if __name__ == '__main__':
    import torchvision.transforms as transforms
    from utils.visual_utils import visual_vector_field

    # t = transforms.Compose([transforms.ToTensor(),
    #                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                              std=[0.229, 0.224, 0.225], inplace=True)])
    t = transforms.ToTensor()
    dataset = Linemod(train=True, transform=t)
    dataloader = torch_utils_data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    for j, data in enumerate(dataloader):
        print("index: ", j)
        color_img, mask_img, coor_map, cls_target, color_img_path = data
        print(f'color shape: {color_img.shape}')
        print(f'mask shape: {mask_img.shape}')
        print(f'coormap shape: {coor_map.shape}')
        print(f'class label: {cls_target}')
        print('class target shape ', cls_target.shape)
        # print(color_img_path)
        # print(mask_img[0][cls_target[0]].sum())
        # if j == 0:
        #     img = np.array(color_img.numpy()[0].transpose([1, 2, 0]) * 255, dtype=np.uint8)[:, :, ::-1]
        #     print(np.sum(img))
        #     cv2.imshow('color', img)
        #     mask_ = mask_img.numpy()[0].astype(np.uint8)
        #     mask_im = mask_ * 255
        #     print(mask_.shape)
        #     cv2.imshow('mask', np.array(mask_im, dtype=np.uint8))
        #
        #     coor_vis = coor_map[0].numpy()[0:2]
        #     print(coor_vis.shape)
        #     field = visual_vector_field(coor_vis)
        #     cv2.imshow('field', field)
        #     cv2.waitKey(-1)
        #     break
