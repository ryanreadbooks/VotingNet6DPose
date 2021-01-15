from typing import Dict

import torch

from nets import VotingNet
from evaluator.evaluators import LinemodEvaluator
from utils.io_utils.inout import save_dict_to_txt


def main():
	# Load the network
	net = VotingNet()
	last_state = torch.load('/home/ryan/Codes/VotingNet6DPose/log_info/models/linemod_cat_fps_debug_epoch500_loss0.056774.pth')
	net.load_state_dict(last_state)
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	net.to(device)
	print('Model loaded into {}, evaluation starts...'.format(device))

	# Define a evaluator for network for cat
	evaluator = LinemodEvaluator(net, refinement=False, simple=True, need_model_origin=False)
	accuracies: Dict = evaluator.evaluate()
	save_dict_to_txt(accuracies, '/content/voting_net_6d/log_info/results/simple_cat_accuracies.txt')
	print(accuracies)


if __name__ == '__main__':
	main()
