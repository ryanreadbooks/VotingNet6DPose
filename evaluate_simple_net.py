import numpy as np
import torch
import torch.nn.functional as F
from nets import VotingNet
from evaluator import VoteProcedure

from utils import draw_linemod_mask_v2, draw_3d_bbox, draw_points, draw_vector_field
from datasets import Linemod, LinemodDatasetProvider
from configs import constants
from PIL import Image
from utils.output_extractor import OutputExtractor, LinemodOutputExtractor
import torch.utils.data as Data
import torchvision.transforms as transforms

if __name__ == '__main__':
    tra = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=constants.IMAGE_MEAN, std=constants.IMAGE_STD)])

    # test_dataset = Linemod(constants.DATASET_PATH_ROOT, train=False, category='cat', dataset_size=1, transform=tra)
    # test_dataloader = Data.DataLoader(test_dataset, batch_size=1, pin_memory=True)

    linemod_dataset = Linemod(root_dir=constants.DATASET_PATH_ROOT,
                              train=False, category='cat',
                              dataset_size=1,
                              transform=tra,
                              need_bg=True,
                              onehot=False,
                              simple=True)

    test_dataloader = Data.DataLoader(linemod_dataset, batch_size=1, pin_memory=True)

    # net = VotingNet()
    net = VotingNet()
    last_state = torch.load('/content/voting_net_6d/log_info/simple_cat_79.pth')
    net.load_state_dict(last_state)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device)
    net.eval()
    print('Model loaded into {}, evaluation starts...'.format(device))

    for data in test_dataloader:
        eval_img, mask_label, vmap_label, label, test_img_path = data
        print('Mask_label shape: ', mask_label.shape)  # shape (1, 480, 640)
        eval_img = eval_img.to(device)
        out = net(eval_img)
        pred_mask: torch.Tensor = out[0]
        pred_vector_map: torch.Tensor = out[1]

        # visualization
        gt_vmap_img = draw_vector_field(vmap_label[0, 0:2].cpu().detach().numpy())
        pred_vmap_img = draw_vector_field(pred_vector_map[0, 0:2].cpu().detach().numpy())
        Image.fromarray(gt_vmap_img, 'RGB').save('/content/voting_net_6d/log_info/gt_vmap_0.png')
        Image.fromarray(pred_vmap_img, 'RGB').save('/content/voting_net_6d/log_info/pred_vmap_0.png')

        # 看一下得出的结果和真实结果之间的损失是多少
        mask_loss = F.cross_entropy(pred_mask, mask_label.type(dtype=torch.long).to(device))
        vector_map_loss = F.smooth_l1_loss(pred_vector_map, vmap_label.to(device))
        print('Loss: mask loss == {:.10f}, vector map loss == {:.6f}'.format(mask_loss.item(), vector_map_loss.item()))
        print('==============================')
        pred_vector_map: np.ndarray = pred_vector_map.cpu().detach().numpy()
        print('Network output mask shape', pred_mask.shape)
        print('Network output vector map shape', pred_vector_map.shape)
        # 每个像素点的概率
        pred_mask = torch.softmax(pred_mask, dim=1)
        # 通道方向取argmax得到预测的每个像素点所属的类别
        binary_mask = pred_mask.argmax(dim=1, keepdim=True)[0, 0]
        print('Binary mask shape', binary_mask.shape)  # shape (480, 640)
        # 将mask二值化用来显示
        binary_mask = torch.where(binary_mask == 0, torch.tensor(255).to(device), torch.tensor(0).to(device))
        binary_mask_np = binary_mask.cpu().detach().numpy().astype(np.uint8)
        # 将二值化的mask保存成图片显示
        Image.fromarray(binary_mask_np, 'L').save('/content/voting_net_6d/log_info/predicted_mask_cat.png')
        # 将gt的mask绘制出来
        gt_mask_label = draw_linemod_mask_v2(mask_label.cpu().numpy()[0])
        # gt_mask_label = draw_linemod_label(mask_label.cpu().numpy()[0])
        gt_mask_label.save('/content/voting_net_6d/log_info/gt_mask.jpg')

        # 尝试投票过程
        # 将属于cat类别的vector map的部分取出来，shape (18, h, w)
        cat_vector_map: np.ndarray = pred_vector_map[0]
        # 构建只有0、1的二值mask，0表示该像素是背景，1表示该像素是物体
        cat_binary_mask = np.where(binary_mask_np == 255, 1, 0)
        # 创建一个投票过程
        the_voting_room = VoteProcedure((480, 640))
        # 进行操作，直接获得关键点位置
        pred_keypoints: np.ndarray = the_voting_room.provide_keypoints(cat_binary_mask, cat_vector_map)
        print('Predicted Keypoints:\n', pred_keypoints)

        # draw the 3d bbox to check
        # 将原始图片读取出来
        test_img_path: str = test_img_path[0]
        test_image = Image.open(test_img_path)
        test_image_points = test_image.copy()
        test_image.save('/content/voting_net_6d/log_info/test_original_image.jpg')
        test_image_with_boxes = draw_3d_bbox(test_image, pred_keypoints[1:], 'blue')
        test_image_label_path = test_img_path.replace('JPEGImages', 'labels')
        test_image_label_path = test_image_label_path.replace('jpg', 'txt')
        gt_keypoints = LinemodDatasetProvider.provide_keypoints_coordinates(test_image_label_path)[1].numpy()
        print('GT keypoints:\n', gt_keypoints)
        test_image_with_boxes = draw_3d_bbox(test_image_with_boxes, gt_keypoints[1:], 'green')
        # 保存结果
        test_image_with_boxes.save('/content/voting_net_6d/log_info/result_image.jpg')

        point_image = draw_points(test_image_points, pred_keypoints, color='blue')
        draw_points(point_image, gt_keypoints, color='green').save('/content/voting_net_6d/log_info/result_points_img.jpg')
