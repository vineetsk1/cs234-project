import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
	parser = argparse.ArgumentParser()

	# Evaluation parameters: which model to run, how often we evaluate, number of times to run for averaging.
	parser.add_argument('-m', '--models', nargs='+', required=True, help='<Required> List of models to run.')
	parser.add_argument('-every', '--print_every', type=int, default=100, help='Print results every <int> epochs during training.')
	parser.add_argument('-n', '--repeats', type=int, default=5, help='Number of times to repeat execution.')
	

	# Data hyperparameters
	parser.add_argument('-im', '--impute_type', type=str2bool, default=False, help='If true, split imputation across races and genders. Otherwise, just do it across entire dataset.')

	# Reward Hyperparameters
	parser.add_argument('-rr', '--real_rewards', type=str2bool, default=False, help='If true, use real rewards for the bandits.')
	parser.add_argument('-rrl2', '--real_rewards_l2', type=str2bool, default=False, help='If using real rewards, use L2 if this is true.')

	# Model-specific hyperparameters
	parser.add_argument('-alpha', '--alpha', type=float, default=0.1, help='<linUCB only> The learning rate.')

	args = parser.parse_args()
	return args