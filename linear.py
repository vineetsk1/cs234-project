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

        self.nsteps = 0
        self.penalty = 1.0

    def _eval(self, xt, thetas=None):
        
        if thetas is None:
            ia1, ia2, ia3 = np.linalg.inv(self.A1), np.linalg.inv(self.A2), np.linalg.inv(self.A3) # Shapes (d, d)
            tt1, tt2, tt3 = ia1.dot(self.b1), ia2.dot(self.b2), ia3.dot(self.b3) # Shapes (d, 1)
            thetas = (ia1, ia2, ia3, tt1, tt2, tt3)
        else:
            ia1, ia2, ia3, tt1, tt2, tt3 = thetas

        pt1 = tt1.T.dot(xt) + self.args.alpha*np.sqrt(xt.T.dot(ia1).dot(xt)) # Shapes (1, 1)
        pt2 = tt2.T.dot(xt) + self.args.alpha*np.sqrt(xt.T.dot(ia2).dot(xt)) # Shapes (1, 1)
        pt3 = tt3.T.dot(xt) + self.args.alpha*np.sqrt(xt.T.dot(ia3).dot(xt)) # Shapes (1, 1)
        pts = [pt1.item(), pt2.item(), pt3.item()]
        pred = np.argmax(pts) + 1 # Arm: 1, 2, or 3
        return pred, thetas, pts

    def train(self, xt, yt):

        self.nsteps += 1
        if self.nsteps % self.args.penalty_after == 0:
            self.penalty *= self.args.time_penalty

        xt = xt.reshape((xt.shape[0], 1)) # Shapes (d, 1)
        pred, _, pts = self._eval(xt)
        if self.args.real_rewards:
            r_t = (pts[pred-1] - pts[yt-1])
            r_t = -abs(r_t) if not self.args.real_rewards_l2 else -(r_t*r_t)
        else:
            if self.args.risk_sensitivity:
                r_t = -1 * abs(pred - yt)
            else:
                r_t = 0 if pred == yt else -1
        r_t *= self.penalty
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
        thetas = None
        for t in range(T):
            pred, thetas, _ = self._eval(X[t].reshape((X[t].shape[0], 1)), thetas)
            preds.append(pred)
        return np.asarray(preds)