import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-da', '--drop_age', default=True, help='Drop rows where age is nan.')
	parser.add_argument('-dh', '--drop_height', default=True, help='Drop rows where height is nan.')
	parser.add_argument('-dw', '--drop_weight', default=True, help='Drop rows where weight is nan.')
	parser.add_argument('-m', '--models', nargs='+', required=True, help='<Required> List of models to run.')
	args = parser.parse_args()
	return args