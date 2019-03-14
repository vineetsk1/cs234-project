import numpy as np
import pandas as pd
from data import get_data
from model import load_model
from tqdm import tqdm

class Evaluator():
    def __init__(self, args, model_name, logger):
        self.args = args
        self.logger = logger

        features, _, labels = get_data(
            args.drop_age, args.drop_height,
            args.drop_weight, args.drop_inr)

        self.model = load_model(model_name, args)

        if not self.model.baseline:
            # Remove features that shouldn't be categorical.
            # Comorbidities has 1436 unique entries
            # Medications has 1840 unique entries
            features = features.drop(['Comorbidities', 'Medications'], axis=1)

            # One hot encode the features, convert to numpy for learning.
            features = pd.get_dummies(features, dummy_na=True).values

        self.features = features
        self.labels = labels
        self.nruns = 0

    def evaluate_model(self):
        self.run_once()

    def run_once(self):

        self.nruns += 1

        # Shuffle order of patients
        X, Y = self.features, self.labels
        inds = np.arange(len(Y))
        np.random.shuffle(inds)
        Y = Y[inds]
        if self.model.baseline:
            X = X.reset_index()
            X = X.reindex(inds)
        else:
            X = X[inds]

        ts, accs, regrets, preds, preds_frozen = [], [], [], [], []
        T, _ = X.shape

        # Training Loop
        if not self.model.baseline:

            self.model.initialize(X)
            for t in tqdm(range(T)):
                pred = self.model.train(X[t], Y[t])
                preds.append(pred)
                if t % self.args.print_every == 0:
                    preds_frozen = self.model.test(X)   # Freeze and test
                    acc_frozen = self.calculate_accuracy(preds_frozen, Y)
                    regret_running = self.calculate_regret(preds, Y[:len(preds)])
                    ts.append(t)
                    accs.append(acc_frozen)            # Report accuracy on entire dataset at each iteration.
                    regrets.append(regret_running)     # But report regret as a "so-far", running metric. See @860.
            preds_frozen = self.model.test(X)
        else:
            preds = self.model.test(X)
            preds_frozen = preds

        # Metrics (Testing)        
        acc_frozen = self.calculate_accuracy(preds_frozen, Y)
        regret_running = self.calculate_regret(preds, Y)
        ts.append(T)
        accs.append(acc_frozen)
        regrets.append(regret_running)

        self.logger.print(self.model.name, "Run", self.nruns)
        self.logger.print(self.model.name, "Timesteps", ts)
        self.logger.print(self.model.name, "Accuracies", accs)
        self.logger.print(self.model.name, "Regrets", regrets)

        return ts, accs, regrets

    def calculate_regret(self, preds, labels):
        incorrect = np.sum(preds != labels)
        return incorrect

    def calculate_accuracy(self, preds, labels):
        total = preds.shape[0]
        correct = np.sum(preds == labels)
        return correct / total