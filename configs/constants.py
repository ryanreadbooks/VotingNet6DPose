"""
some constants configurations are here
"""

import numpy as np

# data root == r'E:\1Downloaded\datasets\LINEMOD_from_yolo-6d'
# data root == '/content/LINEMOD'
# evaluator result path == '/content/voting_net_6d/log_info/results'
# path to the dataset and model saving
DATASET_PATH_ROOT = r'/content/LINEMOD'
MODEL_SAVE_PATH = '/content/voting_net_6d/log_info'
EVALUATOR_RESULTS_PATH = '/content/voting_net_6d/log_info'

# number of classes in linemod dataset
NUM_CLS_LINEMOD = 13
LINEMOD_IMG_HEIGHT = 480
LINEMOD_IMG_WIDTH = 640
LINEMOD_OBJECTS_NAME = ['ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck',
                        'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']

# todo complete the diameter of linemod object
LINEMOD_OBJ_DIAMETER = {
	'ape': 0.0, 'benchvise': 0.0, 'cam': 0.0, 'can': 0.0,
	'cat': 15.2633, 'driller': 0.0, 'duck': 0.0,
	'eggbox': 0.0, 'glue': 0.0, 'holepuncher': 0.0,
	'iron': 0.0, 'lamp': 0.0, 'phone': 0.0
}

NUM_KEYPOINT = 9

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

CAMERA = np.array([[572.4114, 0., 325.2611],
                   [0., 573.57043, 242.04899],
                   [0., 0., 1.]], dtype=np.float32)
