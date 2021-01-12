import torch

from nets import VotingNet
from evaluator.evaluators import LinemodEvaluator


def main():
	net = VotingNet()
	last_state = torch.load('/content/voting_net_6d/log_info/simple_cat_119.pth')
	net.load_state_dict(last_state)
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	net.to(device)
	print('Model loaded into {}, evaluation starts...'.format(device))

	# Define a evaluator for network for cat
	evaluator = LinemodEvaluator(net, 'cat', refinement=False, simple=True)
	pred_pose = evaluator.pipeline(r'/content/LINEMOD/cat/JPEGImages/000002.jpg')
	print(pred_pose)


if __name__ == '__main__':
	main()
