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

    def evaluate_model(self):
        self.run_once()

    def run_once(self):

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

        # Training Loop
        if not self.model.baseline:
            self.model.initialize(X)
            T, _ = X.shape
            for t in tqdm(range(T)):
                self.model.train(X[t], Y[t])

        # Metrics (Testing)
        preds = self.model.test(X)
        acc = self.calculate_accuracy(preds, Y)
        regret = self.calculate_regret(preds, Y)

        self.logger.print(self.model.name, "acc", acc)
        self.logger.print(self.model.name, "regret", regret)

    def calculate_regret(self, preds, labels):
        incorrect = np.sum(preds != labels)
        return incorrect

    def calculate_accuracy(self, preds, labels):
        total = preds.shape[0]
        correct = np.sum(preds == labels)
        return correct / total