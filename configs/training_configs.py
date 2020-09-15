"""
you can configure the training settings here
"""


"""
Indication:
	lr: initial learning rate
	optimizer: type of optimizer, sgd or adam
	batch_size: batch size
	weight_decay: weight decay
	momentum: momentum
	epoch: total epochs for training
	milestone: the epoch at which the model to be saved
	scheduler: use scheduler or not
	scheduler_type: which kind of scheduler to use, 'step_lr', 'multistep_lr', 'exp_lr'
	lr_drop_per_epoch: the learning rate drop every lr_drop_per_epoch epoch
	lr_drop_gamma: the learning rate drop at the rate of lr_drop_gamma according to the scheduler_type
"""
TRAINING_CONFIGS: dict = {
	'lr': 2.5e-4,
	'optimizer': 'sgd',
	'batch_size': 2,
	'weight_decay': 5e-4,
	'momentum': 0.99,
	'epoch': 40,
	'milestone': (10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240),
	'scheduler': True,
	'scheduler_type': 'multistep_lr',
	'lr_drop_per_epoch': 20,    # for step_lr
	'lr_drop_gamma': 0.5
}

TEST_CONFIGS: dict = {
	'batch_size': 1,
}
