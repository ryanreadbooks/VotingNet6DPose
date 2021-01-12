from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class Backbone(nn.Module):
    """
    the backbone of the whole network
    """

    def __init__(self, freeze: bool = True, pretrained: str = 'ignore', local_path=None):
        """
        init function
        :param freeze: need to freeze the backbone or not, default=False
        :param pretrained: need pretrained model or not. It is the resnet backbone that is been pretrained.
        :param local_path: the given path on your machine
        """
        super(Backbone, self).__init__()
        # backbone resnet18
        resnet = None
        if pretrained == 'ignore':
            resnet = resnet18(False)
        elif pretrained == 'local':
            # load the pretrained model params from local machine
            if local_path is not None:
                resnet = resnet18(False)
                pre_trained_params = torch.load(pretrained)
                resnet.load_state_dict(pre_trained_params)
            else:
                raise ValueError('when pretrained is set to local, you need to specify a local path on your machine')
        elif pretrained == 'remote':
            # load the pretrained model params provided by torchvision
            resnet = resnet18(True)
        else:
            raise ValueError('pretrained must be \'ignore\', \'local\', \'remote\'')
        # drop the fc and avgpooling and layer4, because we only need the input to be subsampled to 1/16 of the original size
        resnet.fc = None
        resnet.avgpool = None
        resnet.layer4 = None

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        # by applying this statement, the requires_grad of parameters before it is set to False
        # after it will not be affected
        for p in self.parameters():
            p.requires_grad = not freeze

        self.conv_middle1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv_middle2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x) -> Tuple:
        x = self.conv1(x)
        x = self.bn1(x)
        x2s = self.relu(x)  # go to output, channel = 64
        x = self.maxpool(x2s)
        x4s = self.layer1(x)  # go to output, channel = 64
        x8s = self.layer2(x4s)  # go to output, channel = 128
        x16s = self.layer3(x8s)  # go to output, channel = 256
        x = self.conv_middle1(x16s)
        x = self.relu(x)
        x = self.conv_middle2(x)  # channel = 256
        return x2s, x4s, x8s, x16s, x
