"""
some constants configurations are here
"""

import numpy as np

# data root == r'E:\1Downloaded\datasets\LINEMOD_from_yolo-6d'
# data root == '/content/LINEMOD'
# evaluator result path == '/content/voting_net_6d/log_info/results'
# path to the dataset and model saving
DATASET_PATH_ROOT = r'/content/LINEMOD'
MODEL_SAVE_PATH = '/content/VotingNet6DPose/log_info'
EVALUATOR_RESULTS_PATH = '/content/VotingNet6DPose/log_info'

# number of classes in linemod dataset
NUM_CLS_LINEMOD = 13
LINEMOD_IMG_HEIGHT = 480
LINEMOD_IMG_WIDTH = 640
LINEMOD_OBJECTS_NAME = ['ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck',
                        'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']

# todo complete the diameter of linemod object
LINEMOD_OBJ_DIAMETER = {
	'ape': 9.74298,
	'benchvise': 28.6908,
	'cam': 17.1593,
	'can': 19.3416,
	'cat': 15.2633,
	'driller': 25.9425,
	'duck': 10.7131,
	'eggbox': 17.6364,
	'glue': 16.4857,
	'holepuncher': 14.8204,
	'iron': 30.3153,
	'lamp': 28.5155,
	'phone': 20.8394
}

NUM_KEYPOINT = 8

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

CAMERA = np.array([[572.4114, 0., 325.2611],
                   [0., 573.57043, 242.04899],
                   [0., 0., 1.]], dtype=np.float64)
