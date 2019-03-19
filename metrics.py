import numpy as np
import pandas as pd
from data import get_data
from model import load_model
from tqdm import tqdm
from utils import args_to_str

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

class Evaluator():
    def __init__(self, args, model_name, logger, directory="plots"):
        self.args = args
        self.logger = logger

        features, _, labels = get_data(args)

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
        self.logger.print("Data has {} labels".format(len(labels)))
        self.nruns = 0

        self.dir = directory
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def make_plot(self, x, ymean, title, name, yerr=[]):
        sns.set_style("darkgrid")
        if len(yerr) == 0:
            plt.plot(x, ymean, marker='.')
        else:
            plt.errorbar(x, ymean, yerr=yerr, fmt='.', linestyle='-')
        plt.title(title)

        pre = "{}_{}_run_{}_".format(self.model.name, name, self.nruns if len(yerr) == 0 else "all")
        fname = args_to_str(self.args, ext=".png", pre=pre)
        self.logger.print("Saving plot...", fname)
        plt.savefig(os.path.join(self.dir, fname))
        plt.close()

    def evaluate_model(self):

        all_ts = []
        all_accs = []
        all_regrets = []

        for _ in range(self.args.repeats):
            ts, accs, regrets = self.run_once()
            all_ts.append(ts)
            all_accs.append(accs)
            all_regrets.append(regrets)
            self.make_plot(ts, accs, "Accuracy vs. Time", "acc")
            self.make_plot(ts, regrets, "Regret vs. Time", "regret")

        self.logger.print(self.model.name, "Final Stats")
        self.logger.print(self.model.name, "Timesteps", all_ts)
        self.logger.print(self.model.name, "Accuracies", all_accs)
        self.logger.print(self.model.name, "Regrets", all_regrets)

        all_ts = np.asarray(all_ts)
        all_accs = np.asarray(all_accs)
        all_regrets = np.asarray(all_regrets)

        self.logger.print(self.model.name, "Average")
        self.logger.print(self.model.name, "Timesteps", all_ts.mean(axis=0))
        self.logger.print(self.model.name, "Accuracies", all_accs.mean(axis=0))
        self.logger.print(self.model.name, "Regrets", all_regrets.mean(axis=0))

        self.make_plot(all_ts.mean(axis=0), all_accs.mean(axis=0), "Accuracy vs. Time", "acc", 2*all_accs.std(axis=0))
        self.make_plot(all_ts.mean(axis=0), all_regrets.mean(axis=0), "Regret vs. Time", "regret", 2*all_regrets.std(axis=0))

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