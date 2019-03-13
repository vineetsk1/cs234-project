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
	parser.add_argument('-da', '--drop_age', type=str2bool, default=True, help='Drop rows where age is nan.')
	parser.add_argument('-dh', '--drop_height', type=str2bool, default=True, help='Drop rows where height is nan.')
	parser.add_argument('-dw', '--drop_weight', type=str2bool, default=True, help='Drop rows where weight is nan.')
	parser.add_argument('-di', '--drop_inr', type=str2bool, default=False, help='Drop rows where target inr is nan.')
	parser.add_argument('-alpha', '--alpha', type=float, default=0.1, help='<linUCB only> The learning rate.')
	parser.add_argument('-m', '--models', nargs='+', required=True, help='<Required> List of models to run.')
	args = parser.parse_args()
	return args