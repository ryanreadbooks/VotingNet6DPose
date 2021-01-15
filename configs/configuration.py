import os
import yaml
import numpy as np

_legal_categories = ['ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck',
                     'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']
_legal_keypoint_types = ['fps', 'corners']
_legal_optimizer = ['sgd', 'adam']
_legal_scheduler = ['step', 'multistep', 'exponential']


class RegularConfig:
    def __init__(self):
        abs_file_path: str = os.path.dirname(__file__)
        self.project_dir = abs_file_path.replace('configs', '')
        root_path: str = os.path.split(abs_file_path)[0]
        config_file_path: str = os.path.join(root_path, 'configs.yml')
        f = open(config_file_path, 'r')

        overall_config = yaml.load(f, Loader=yaml.SafeLoader)
        path_config = overall_config['path']
        camera_config = overall_config['camera']

        self.category = overall_config['category'].lower()
        if self.category not in _legal_categories:
            raise ValueError('invalid value for category')

        self.num_keypoint = overall_config['num-keypoint']
        self.keypoint_type = overall_config['keypoint-type'].lower()
        if self.keypoint_type not in _legal_keypoint_types:
            raise ValueError('Invalid keypoint type!')
        if self.keypoint_type == 'fps' and not self.num_keypoint == 8:
            raise ValueError('when using fps as keypoints, the NUM_KEYPOINT must be 8')
        if self.keypoint_type == 'corners' and self.keypoint_type == 9:
            raise ValueError('when using corners as keypoints, the NUM_KEYPOINT must be 9')

        self.mode = overall_config['mode']
        self.data_path_root = path_config['data-path-root']
        self.model_saved_path = path_config['model-saved-path']
        self.result_path = path_config['result-path']

        self.dataset_name = overall_config['dataset-name']
        self.num_of_class = overall_config['num-of-class']

        self.img_width = overall_config['img-width']
        self.img_height = overall_config['img-height']

        fx, fy, = camera_config['fx'], camera_config['fy']
        cx, cy = camera_config['cx'], camera_config['cy']
        self.camera = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]], dtype=np.float64)

        transform_config = overall_config['random-transform']
        self.random_rotate_angle = transform_config['rotate-angle']
        self.random_scale = transform_config['scale']
        self.random_translation = transform_config['translation']

        f.close()


class TrainingConfig:

    def __init__(self):
        abs_file_path: str = os.path.dirname(__file__)
        root_path: str = os.path.split(abs_file_path)[0]
        config_file_path: str = os.path.join(root_path, 'configs.yml')
        f = open(config_file_path, 'r')
        config = yaml.load(f, Loader=yaml.SafeLoader)['training']

        self.lr = config['lr']
        self.optimizer = config['optimizer'].lower()
        if self.optimizer not in _legal_optimizer:
            raise ValueError('Invalid optimizer!')
        self.batch_size = config['batch-size']
        self.weight_decay = config['weight-decay']
        self.momentum = config['momentum']
        self.epochs = config['epochs']
        self.cuda = config['cuda']
        milestones_str: str = config['milestones'][1:-1]
        self.milestones = [int(i) for i in milestones_str.split(',')]
        self.log_train = config['log-train']
        self.scheduler = config['scheduler'].lower()
        if self.scheduler != '':
            if self.scheduler not in _legal_scheduler:
                raise ValueError('Invalid scheduler type!')
        self.frequency = config['lr-drop']['freq']
        self.gamma = config['lr-drop']['gamma']

        f.close()
        # self.display()

    def display(self):
        print('-------------Training configurations ----------------')
        print('lr', self.lr)
        print('optimizer', self.optimizer)
        print('batch_size', self.batch_size)
        print('weight_decay', self.weight_decay)
        print('momentum', self.momentum)
        print('epochs', self.epochs)
        print('cuda', self.cuda)
        print('log_train', self.log_train)
        print('scheduler', self.scheduler)
        print('frequency', self.frequency)
        print('gamma', self.gamma)
        print('-----------------------------------------------------')


train_config = TrainingConfig()
regular_config = RegularConfig()
