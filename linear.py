import numpy as np
import pandas as pd
from baseline import Model

class LinearUCB(Model):
    def __init__(self, name, args, logger):
        super().__init__(name, args, logger)

    def run(self, features, labels):

        # Remove features that shouldn't be categorical.
        # Comorbidities has 1436 unique entries
        # Medications has 1840 unique entries
        features = features.drop(['Comorbidities', 'Medications'], axis=1)

        # Debug: To print out number of unique values in each column
        # Helps see which categorical variables shouldn't be categorical.
        # for column in features:
        #     nunique = len(features[column].unique())
        #     print(column, nunique)

        # One hot encode the features
        features = pd.get_dummies(features, dummy_na=True).values

        # Shuffle order of patients        
        inds = np.arange(len(features))
        np.random.shuffle(inds)
        features = features[inds]
        labels = labels[inds]

        # T = number of rounds (number of patients)
        # d = dimensionality of features
        # k = number of arms = 3
        T, d = features.shape
        alpha = self.args.alpha

        # Assign a differerent weight/bias (A, b) for each arm
        A1, A2, A3 = np.identity(d), np.identity(d), np.identity(d)
        b1, b2, b3 = np.zeros(d), np.zeros(d), np.zeros(d)

        preds = []

        for t in range(T):
            xt = features[t]
            
            ia1, ia2, ia3 = np.linalg.inv(A1), np.linalg.inv(A2), np.linalg.inv(A3)

            tt1, tt2, tt3 = ia1.dot(b1), ia2.dot(b2), ia3.dot(b3)
            pt1 = tt1.T.dot(xt) + alpha*np.sqrt(xt.T.dot(ia1).dot(xt))
            pt2 = tt2.T.dot(xt) + alpha*np.sqrt(xt.T.dot(ia2).dot(xt))
            pt3 = tt3.T.dot(xt) + alpha*np.sqrt(xt.T.dot(ia3).dot(xt))

            pred = np.argmax([pt1, pt2, pt3]) + 1 # Arm: 1, 2, or 3
            r_t = 0 if pred == labels[t] else -1
            # self.logger.print("Probabilities:", pt1, pt2, pt3)
            # self.logger.print("Chose arm", pred, "reward", r_t)

            if pred == 1:
                A1 = A1 + xt.T.dot(xt)
                b1 = b1 + xt.dot(r_t)
            elif pred == 2:
                A2 = A2 + xt.T.dot(xt)
                b2 = b2 + xt.dot(r_t)
            elif pred == 3:
                A3 = A3 + xt.T.dot(xt)
                b3 = b3 + xt.dot(r_t)

            preds.append(pred)
        
        preds = np.asarray(preds)
        return preds, labels