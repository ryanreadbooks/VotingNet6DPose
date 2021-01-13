"""
@ Author: ryanreadbooks
@ Time: 2021/1/12
@ File name: data_augmentation.py
@ File description: The implementation of data augmentation, including rotation, translation and scaling
"""

import numpy as np
import cv2
import random


def rotate_im(img: np.ndarray, kps: np.ndarray, angle: float):
    """
    Rotate the image along with the keypoint coordinates
    :param img: image，in the format of opencv ndarray
    :param kps: keypoints coordinates, shape of (n, 2) with n to be the number of keypoint
    :param angle: the rotation angle in degree
    :return: the rotated image and the rotated keypoint
    """
    width, height = img.shape[1], img.shape[0]
    (cX, cY) = (width // 2, height // 2)
    # get the rotation matrix [R|t] shape of (2, 3)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1)
    # rotate the image
    rotated_img = cv2.warpAffine(img, M, (width, height))
    # todo handle the situation of keypoint out of bound
    rotation_mat = M[:, :2]
    translation_vec = M[:, -1].reshape((-1, 1))
    rotated_kps = (rotation_mat @ kps.T).T
    rotated_kps[:, 0] += translation_vec[0]
    rotated_kps[:, 1] += translation_vec[1]

    return rotated_img, rotated_kps


def translate_im(img, kps, tx, ty):
    """
    Translate the image along with the keypoint coordinates
    :param img: image，in the format of opencv ndarray
    :param kps: keypoints coordinates, shape of (n, 2) with n to be the number of keypoint
    :param tx: the translation pixels along x direction
    :param ty: the translation pixels along y direction
    :return: the translated image and the translated keypoint
    """
    width, height = img.shape[1], img.shape[0]
    # get rotation matrix，2x3 [R|t]，regard the translation to be a rotation without rotating
    M = np.array([[1, 0, tx],
                  [0, 1, ty]], dtype=np.float32)
    # translate the image
    translated_img = cv2.warpAffine(img, M, (width, height))
    # odo handle the situation of keypoint out of bound
    rotation_mat = M[:, :2]
    translation_vec = M[:, -1].reshape((-1, 1))
    translated_kps = (rotation_mat @ kps.T).T
    translated_kps[:, 0] += translation_vec[0]
    translated_kps[:, 1] += translation_vec[1]

    return translated_img, translated_kps


def random_scale_im(img, kps, scale=0.2):
    """
    Randomly scale the image
    :param img: image，in the format of opencv ndarray
    :param kps: keypoints coordinates, shape of (n, 2) with n to be the number of keypoint
    :param scale: the scale is performed by a factor chosen from (1-scale, 1+scale)
    :return: the scaled image and the scaled keypoint
    """

    img_shape = img.shape
    width, height = img.shape[1], img.shape[0]
    scale = (max(-1, -scale), scale)
    scale_x = random.uniform(*scale)
    scale_y = random.uniform(*scale)

    resize_scale_x = 1 + scale_x
    resize_scale_y = 1 + scale_y

    img = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)

    # todo handle the out-of-bound situation
    kps_scaled = np.copy(kps)
    kps_scaled[:, 0] *= resize_scale_x
    kps_scaled[:, 1] *= resize_scale_y

    canvas = np.zeros(img_shape, dtype=np.uint8)

    y_lim = int(min(resize_scale_y, 1) * img_shape[0])
    x_lim = int(min(resize_scale_x, 1) * img_shape[1])

    canvas[:y_lim, :x_lim, :] = img[:y_lim, :x_lim, :]

    img = canvas

    return img, kps_scaled


class RandomTransform:

    def __call__(self, img, mask, keypoints):
        """
        Perform random transformation, including random rotation, random translation and random scaling.
        @param img: the image to be transformed
        @param mask:
        @param keypoints:
        @return: the transformed image
        """
