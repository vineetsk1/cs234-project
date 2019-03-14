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
        A1, A2, A3 = np.identity(d), np.identity(d), np.identity(d) # Shapes (d, d)
        b1, b2, b3 = np.zeros((d, 1)), np.zeros((d, 1)), np.zeros((d, 1)) # Shapes (d, 1)

        preds = []

        for t in range(T):
            xt = features[t].reshape((d, 1)) # Shapes (d, 1)
            
            ia1, ia2, ia3 = np.linalg.inv(A1), np.linalg.inv(A2), np.linalg.inv(A3) # Shapes (d, d)

            tt1, tt2, tt3 = ia1.dot(b1), ia2.dot(b2), ia3.dot(b3) # Shapes (d, 1)
            pt1 = tt1.T.dot(xt) + alpha*np.sqrt(xt.T.dot(ia1).dot(xt)) # Shapes (1, 1)
            pt2 = tt2.T.dot(xt) + alpha*np.sqrt(xt.T.dot(ia2).dot(xt)) # Shapes (1, 1)
            pt3 = tt3.T.dot(xt) + alpha*np.sqrt(xt.T.dot(ia3).dot(xt)) # Shapes (1, 1)

            pred = np.argmax([pt1, pt2, pt3]) + 1 # Arm: 1, 2, or 3
            r_t = 0 if pred == labels[t] else -1

            if pred == 1:
                A1 = A1 + xt.dot(xt.T) # Shapes (d, d)
                b1 = b1 + r_t * xt     # Shapes (d, 1)
            elif pred == 2:
                A2 = A2 + xt.dot(xt.T) # Shapes (d, d)
                b2 = b2 + r_t * xt     # Shapes (d, 1)
            elif pred == 3:
                A3 = A3 + xt.dot(xt.T) # Shapes (d, d)
                b3 = b3 + r_t * xt     # Shapes (d, 1)

            preds.append(pred)
        
        preds = np.asarray(preds)
        return preds, labels