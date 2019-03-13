from data import get_data
from model import load_model
from utils import evaluate_preds
from parse_args import parse_args
from create_logger import Logger

def main():

    args = parse_args()
    log = Logger(args)
    features, _, labels = get_data(
        args.drop_age, args.drop_height,
        args.drop_weight, args.drop_inr)
    
    for model_name in args.models:
        model = load_model(model_name, args, log)
        preds, lbls = model.run(features, labels)
        acc = evaluate_preds(preds, lbls)
        log.print(model.name, "acc", acc)

    log.close()

if __name__ == "__main__":
    main()