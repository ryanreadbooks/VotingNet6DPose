"""
@ Author: ryanreadbooks
@ Time: 2021/1/12
@ File name: data_augmentation.py
@ File description: The implementation of data augmentation, including rotation, translation and scaling
"""

import random
import numpy as np
import cv2
from PIL import Image

from configs.configuration import regular_config
from utils.visual_utils import draw_points, draw_3d_bbox


def rotate_im(img: np.ndarray, mask: np.ndarray, kps: np.ndarray, angle: float):
    """
    Rotate the image along with the keypoint coordinates
    :param img: image，in the format of opencv ndarray
    :param mask: mask to be rotated
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
    rotated_mask = cv2.warpAffine(mask, M, (width, height))
    rotation_mat = M[:, :2]
    translation_vec = M[:, -1].reshape((-1, 1))
    rotated_kps = (rotation_mat @ kps.T).T
    rotated_kps[:, 0] += translation_vec[0]
    rotated_kps[:, 1] += translation_vec[1]

    return rotated_img, rotated_mask, rotated_kps


def translate_im(img, mask, kps, tx, ty):
    """
    Translate the image along with the keypoint coordinates
    :param img: image，in the format of opencv ndarray
    :param mask: mask
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
    translated_mask = cv2.warpAffine(mask, M, (width, height))
    # odo handle the situation of keypoint out of bound
    rotation_mat = M[:, :2]
    translation_vec = M[:, -1].reshape((-1, 1))
    translated_kps = (rotation_mat @ kps.T).T
    translated_kps[:, 0] += translation_vec[0]
    translated_kps[:, 1] += translation_vec[1]

    return translated_img, translated_mask, translated_kps


def random_scale_im(img, mask, kps, scale=0.2):
    """
    Randomly scale the image
    :param img: image，in the format of opencv ndarray
    :param mask: mask
    :param kps: keypoints coordinates, shape of (n, 2) with n to be the number of keypoint
    :param scale: the scale is performed by a factor chosen from (1-scale, 1+scale)
    :return: the scaled image and the scaled keypoint
    """

    img_shape = img.shape
    mask_shape = mask.shape
    scale = (max(-1, -scale), scale)
    scale_x = random.uniform(*scale)
    scale_y = random.uniform(*scale)

    resize_scale_x = 1 + scale_x
    resize_scale_y = 1 + scale_y

    img = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)
    mask = cv2.resize(mask, None, fx=resize_scale_x, fy=resize_scale_y)

    kps_scaled = np.copy(kps)
    kps_scaled[:, 0] *= resize_scale_x
    kps_scaled[:, 1] *= resize_scale_y

    canvas = np.zeros(img_shape, dtype=np.uint8)
    canvas_mask = np.zeros(mask_shape, dtype=np.uint8)

    y_lim = int(min(resize_scale_y, 1) * img_shape[0])
    x_lim = int(min(resize_scale_x, 1) * img_shape[1])

    canvas[:y_lim, :x_lim, :] = img[:y_lim, :x_lim, :]
    canvas_mask[:y_lim, :x_lim] = mask[:y_lim, :x_lim]

    img = canvas
    mask = canvas_mask

    return img, mask, kps_scaled


class RandomTransform:

    def __call__(self, img: Image.Image, mask: np.ndarray, keypoints: np.ndarray):
        """
        Perform random transformation, including random rotation, random translation and random scaling.
        @param img: the image to be transformed
        @param mask: the corresponding mask
        @param keypoints: the corresponding keypoints
        @return: the transformed image
        """

        angle = regular_config.random_rotate_angle
        tx = regular_config.random_translation
        ty = tx // 2
        scale_factor = regular_config.random_scale
        width = regular_config.img_width
        height = regular_config.img_height

        # pick a angle from (-angle, angle)
        angle = random.randint(-angle, angle)
        tx = random.randint(-tx, tx)
        ty = random.randint(-ty, ty)

        img_arr = np.asanyarray(img)[:, :, ::-1]
        img_arr, mask, keypoints = rotate_im(img_arr, mask, keypoints, angle)
        img_arr, mask, keypoints = translate_im(img_arr, mask, keypoints, tx, ty)
        img_arr, mask, keypoints = random_scale_im(img_arr, mask, keypoints, scale_factor)

        keypoints_cp = keypoints.copy()
        correction_x, correction_y = 0, 0

        if sum(keypoints_cp[:, 0] <= 0):
            # cross x border
            correction_x = width // 3
        if sum(keypoints_cp[:, 0] >= width):
            # cross x border
            correction_x = -width // 3
        if sum(keypoints_cp[:, 1] <= 0):
            # cross y border
            correction_y = height // 3
        if sum(keypoints_cp[:, 1] >= height):
            # cross y border
            correction_y = -height // 3
        # translate back into the image region
        translate_im(img_arr, mask, keypoints, correction_x, correction_y)

        # cv2.imshow('img_arr', img_arr)
        # mask = np.where(mask == 1, 255, 0).astype(np.uint8)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(-1)

        img_transformed = Image.fromarray(img_arr[:, :, ::-1], 'RGB')
        # draw_points(img_transformed, keypoints).show()
        # draw_3d_bbox(img_transformed, keypoints[1:, :]).show()

        return img_transformed, mask, keypoints
