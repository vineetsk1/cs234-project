import numpy as np
import pandas as pd
from baseline import Model

class LinearUCB(Model):
    def __init__(self, name, args):
        super().__init__(name, args, False)

    def initialize(self, features):
        _, d = features.shape
        self.d = d

        # Assign a differerent weight/bias (A, b) for each arm
        self.A1, self.A2, self.A3 = np.identity(d), np.identity(d), np.identity(d)       # Shapes (d, d)
        self.b1, self.b2, self.b3 = np.zeros((d, 1)), np.zeros((d, 1)), np.zeros((d, 1)) # Shapes (d, 1)

    def _eval(self, xt, ia1=None, ia2=None, ia3=None, tt1=None, tt2=None, tt3=None):
        
        if ia1 is None:
            ia1, ia2, ia3 = np.linalg.inv(self.A1), np.linalg.inv(self.A2), np.linalg.inv(self.A3) # Shapes (d, d)
            tt1, tt2, tt3 = ia1.dot(self.b1), ia2.dot(self.b2), ia3.dot(self.b3) # Shapes (d, 1)

        pt1 = tt1.T.dot(xt) + self.args.alpha*np.sqrt(xt.T.dot(ia1).dot(xt)) # Shapes (1, 1)
        pt2 = tt2.T.dot(xt) + self.args.alpha*np.sqrt(xt.T.dot(ia2).dot(xt)) # Shapes (1, 1)
        pt3 = tt3.T.dot(xt) + self.args.alpha*np.sqrt(xt.T.dot(ia3).dot(xt)) # Shapes (1, 1)

        pred = np.argmax([pt1, pt2, pt3]) + 1 # Arm: 1, 2, or 3
        return pred, (ia1, ia2, ia3, tt1, tt2, tt3)

    def train(self, xt, yt):
        xt = xt.reshape((xt.shape[0], 1)) # Shapes (d, 1)
        pred, _ = self._eval(xt)
        r_t = 0 if pred == yt else -1
        if pred == 1:
            self.A1 = self.A1 + xt.dot(xt.T) # Shapes (d, d)
            self.b1 = self.b1 + r_t * xt    # Shapes (d, 1)
        elif pred == 2:
            self.A2 = self.A2 + xt.dot(xt.T) # Shapes (d, d)
            self.b2 = self.b2 + r_t * xt    # Shapes (d, 1)
        elif pred == 3:
            self.A3 = self.A3 + xt.dot(xt.T) # Shapes (d, d)
            self.b3 = self.b3 + r_t * xt    # Shapes (d, 1)
        return pred

    def test(self, X):
        T, _ = X.shape
        preds = []
        ia1, ia2, ia3, tt1, tt2, tt3 = None, None, None, None, None, None
        for t in range(T):
            pred, invs = self._eval(X[t].reshape((X[t].shape[0], 1)), ia1, ia2, ia3, tt1, tt2, tt3)
            ia1, ia2, ia3, tt1, tt2, tt3 = invs
            preds.append(pred)
        return np.asarray(preds)