from parse_args import parse_args
from create_logger import Logger
from metrics import Evaluator

def main():

    args = parse_args()
    logger = Logger(args)
    
    for model_name in args.models:
        evaluator = Evaluator(args, model_name, logger)
        evaluator.evaluate_model()

    logger.close()

if __name__ == "__main__":
    main()