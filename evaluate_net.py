from typing import List, Tuple

import numpy as np
import cv2 as cv
import torch
from torch.backends import cudnn
import torch.nn.functional as F
from nets import VotingNet
from evaluator import VoteProcedure

from utils import draw_linemod_mask_v2, draw_3d_bbox, draw_points, draw_vector_field
from datasets import Linemod, LinemodDatasetProvider
from configs import constants
from configs.configuration import regular_config
from PIL import Image
from utils.output_extractor import OutputExtractor, LinemodOutputExtractor
import torch.utils.data as Data
import torchvision.transforms as transforms

if __name__ == '__main__':
    cudnn.benchmark = True
    cudnn.deterministic = True
    tra = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=constants.IMAGE_MEAN, std=constants.IMAGE_STD)])

    # test_dataset = Linemod(constants.DATASET_PATH_ROOT, train=False, category='cat', dataset_size=1, transform=tra)
    # test_dataloader = Data.DataLoader(test_dataset, batch_size=1, pin_memory=True)

    linemod_dataset = Linemod(train=False, transform=tra)

    test_dataloader = Data.DataLoader(linemod_dataset, batch_size=1, pin_memory=True)

    # net = VotingNet()
    net = VotingNet()
    last_state = torch.load('/home/ryan/Codes/VotingNet6DPose/log_info/models/linemod_cat_fps_debug_epoch500_loss0.056774.pth')
    net.load_state_dict(last_state)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device)
    net.eval()
    print('Model loaded into {}, evaluation starts...'.format(device))

    for i, data in enumerate(test_dataloader):
        eval_img, mask_label, vmap_label, label, test_img_path = data
        print('Mask_label shape: ', mask_label.shape)  # shape (1, 480, 640)
        eval_img = eval_img.to(device)
        out = net(eval_img)
        pred_mask: torch.Tensor = out[0]
        pred_vector_map: torch.Tensor = out[1]

        pred_mask, pred_vector_map = pred_mask.cpu().detach().numpy(), pred_vector_map.cpu().detach().numpy()

        # find center of the mask
        mask = np.where(pred_mask >= 0.5, 1, 0).astype(np.uint8)
        population = np.where(mask[0][0] >= 1)[:2]
        population: List[Tuple] = list(zip(population[1], population[0]))  # the set of coordinates, format List[(x,y)]
        center = np.asarray(population).mean(axis=0)
        center_x, center_y = int(center[0]), int(center[1])
        region_x = 100
        region_y = 100
        kps_from_img_process = list()
        # visualization
        region = 20
        max_detected_kp = 10
        for k in range(regular_config.num_keypoint):
            # print(vmap_label[0][k * 2: (k + 1) * 2].shape)
            # print(pred_vector_map[0][k * 2: (k + 1) * 2].shape)
            gt_vmap_img = draw_vector_field(vmap_label[0][k * 2: (k + 1) * 2])
            pred_vmap_img = draw_vector_field(pred_vector_map[0][k * 2: (k + 1) * 2])
            #
            # # 截取mask附近的区域出来
            # region_start_y = max(0, center_y - region_y)
            # region_start_x = max(0, center_x - region_x)
            # region_pred = pred_vmap_img[region_start_y: min(center_y + region_y, 480 - 1), region_start_x: min(center_x + region_x, 640 - 1)]
            # # 获得可能的关键点
            # track = cv.goodFeaturesToTrack(cv.cvtColor(region_pred, cv.COLOR_RGB2GRAY), max_detected_kp, 0.05, 10)
            # try:
            #     kps = track.reshape(max_detected_kp, 2)
            # except:
            #     kps = track
            #     pass
            # # 对关键点进行筛选
            # # 这里的kp是在小区域里面的坐标， 还要转换成在原始坐标系里面的坐标
            # for kp in kps:
            #     has_color = set()
            #     # 附近10个pixels的邻域
            #     kp_x, kp_y = int(kp[0]), int(kp[1])
            #     neighborhood = region_pred[kp_y - region: kp_y + region, kp_x - region: kp_x + region]
            #     # 判断是否有8种颜色在邻域中，如果是，则认为是关键点
            #     height, width = neighborhood.shape[0], neighborhood.shape[1]
            #     for v in range(height):
            #         for u in range(width):
            #             color = tuple(neighborhood[v, u, :])
            #             has_color.add(color)
            #     print('length of kps:', len(has_color), kp)
            #     if len(has_color) == 8:
            #         kp[0] += region_start_x
            #         kp[1] += region_start_y
            #         print(kp)
            #         kps_from_img_process.append(kp)
            #         has_color.clear()
            #         break
            Image.fromarray(gt_vmap_img, 'RGB').save('/home/ryan/Codes/VotingNet6DPose/log_info/results/gt_vmap_{:d}.png'.format(k))
            Image.fromarray(pred_vmap_img, 'RGB').save('/home/ryan/Codes/VotingNet6DPose/log_info/results/pred_vmap_{:d}.png'.format(k))

        # kps_from_img_process_nparr = np.asarray(kps_from_img_process)
        # print('kps_from_img_process: \n', kps_from_img_process_nparr)

        # 看一下得出的结果和真实结果之间的损失是多少
        # print('pred_mask.shape', pred_mask.shape)
        # print('mask_label.shape', mask_label.shape)
        # print('pred_vector_map.shape', pred_vector_map.shape)
        # print('vmap_label.shape', vmap_label.shape)
        # mask_loss = F.cross_entropy(pred_mask, mask_label.type(dtype=torch.long).to(device))
        # vector_map_loss = F.smooth_l1_loss(pred_vector_map, vmap_label.to(device))
        # print('Loss: mask loss == {:.10f}, vector map loss == {:.6f}'.format(mask_loss.item(), vector_map_loss.item()))
        print('==============================')
        # pred_vector_map: np.ndarray = pred_vector_map.detach()
        # print('Network output mask shape', pred_mask.shape)
        # print('Network output vector map shape', pred_vector_map.shape)
        # 每个像素点的概率
        # pred_mask = torch.softmax(pred_mask, dim=1)
        # 通道方向取argmax得到预测的每个像素点所属的类别
        binary_mask = np.where(pred_mask >= 0.5, 255, 0).astype(np.uint8)
        print('Binary mask shape', binary_mask.shape)  # shape (480, 640)
        # 将mask二值化用来显示
        # binary_mask = torch.where(binary_mask == torch.tensor(0.0, device=device), torch.tensor(255).to(device), torch.tensor(0).to(device))
        # binary_mask_np = binary_mask.cpu().detach().numpy().astype(np.uint8)
        # 将二值化的mask保存成图片显示
        Image.fromarray(binary_mask[0][0], 'L').save('/home/ryan/Codes/VotingNet6DPose/log_info/results/predicted_mask_cat.png')
        # 将gt的mask绘制出来
        gt_mask_label = draw_linemod_mask_v2(mask_label.cpu().numpy()[0])
        # gt_mask_label = draw_linemod_label(mask_label.cpu().numpy()[0])
        gt_mask_label.save('/home/ryan/Codes/VotingNet6DPose/log_info/results/gt_mask.jpg')

        # 尝试投票过程
        # 将属于cat类别的vector map的部分取出来，shape (18, h, w)
        cat_vector_map: np.ndarray = pred_vector_map[0]
        # 构建只有0、1的二值mask，0表示该像素是背景，1表示该像素是物体
        cat_binary_mask = np.where(binary_mask == 255, 1, 0)
        # 创建一个投票过程
        the_voting_room = VoteProcedure((480, 640))
        # 进行操作，直接获得关键点位置
        pred_keypoints: np.ndarray = the_voting_room.provide_keypoints(cat_binary_mask, cat_vector_map, 0.9, True)
        print('Predicted Keypoints:\n', pred_keypoints)

        # draw the 3d bbox to check
        # 将原始图片读取出来
        print(test_img_path)
        print(type(test_img_path))
        test_img_path: str = test_img_path[0]
        test_image = Image.open(test_img_path)
        test_image_points = test_image.copy()
        test_image.save('/home/ryan/Codes/VotingNet6DPose/log_info/results/test_original_image.jpg')
        # test_image_with_boxes = draw_3d_bbox(test_image, pred_keypoints[1:], 'blue')
        test_image_label_path = test_img_path.replace('JPEGImages', 'labels')
        test_image_label_path = test_image_label_path.replace('jpg', 'txt')
        gt_keypoints = LinemodDatasetProvider.provide_keypoints_coordinates(test_image_label_path)[1].numpy()
        print('GT keypoints:\n', gt_keypoints)
        # test_image_with_boxes = draw_3d_bbox(test_image_with_boxes, gt_keypoints[1:], 'green')
        # 保存结果
        # test_image_with_boxes.save('/home/ryan/Codes/VotingNet6DPose/log_info/results/result_image.jpg')

        point_image = draw_points(test_image_points, pred_keypoints, color='blue')
        draw_points(point_image, gt_keypoints, color='green').save('/home/ryan/Codes/VotingNet6DPose/log_info/results/result_points_img.jpg')

        if i == 0:
            break
