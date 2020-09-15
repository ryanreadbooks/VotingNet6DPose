from typing import Dict

import torch

from nets import VotingNetWithBg
from evaluator.evaluators import LinemodEvaluator
from utils.io_utils.inout import save_dict_to_txt


def main():
	# Load the network
	net = VotingNetWithBg()
	last_state = torch.load('/content/voting_net_6d/log_info/finetune_cat_79_0907_ok.pth')
	net.load_state_dict(last_state)
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	net.to(device)
	print('Model loaded into {}, evaluation starts...'.format(device))

	# Define a evaluator for network for cat
	evaluator = LinemodEvaluator(net, 'cat', refinement=False)
	accuracies: Dict = evaluator.evaluate()
	save_dict_to_txt(accuracies, 'log_info/results/cat_accuracies.txt')
	print(accuracies)


if __name__ == '__main__':
	main()
