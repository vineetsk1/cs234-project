import numpy as np
import pandas as pd
from baseline import Model

from sklearn.linear_model import Lasso

class LassoBandit(Model):
    def __init__(self, name, args):
        super().__init__(name, args, False)

    def initialize(self, features):

        T, d = features.shape
        q, l1, l2, K = self.args.q, self.args.lambda1, self.args.lambda2, 3
        self.T, self.d, self.K = T, d, K
        self.l1, self.l2, self.l2_orig = l1, l2, l2

        self.BS = {}
        for i in range(1, K+1):
            self.BS[i] = Lasso(alpha=self.l2, fit_intercept=False, warm_start=True)
            self.BS["{}_trained".format(i)] = False

        self.Tset = {}
        self.t = 0
        for i in range(1, K+1):
            self.Tset[i] = set()
            done = False
            for n in range(0, T):
                for j in range(q*i - q + 1, q*i + 1):
                    ti = (pow(2, n) - 1)*K*q + j
                    if ti > T and j == q*i - q + 1: # First j value didn't work, so n has grown too large.
                        done = True
                        break
                    self.Tset[i].add(ti)
                if done: break

    def train(self, xt, yt):
        self.t += 1

        xt = xt.reshape((1, xt.shape[0])) # Shape (1, d)
        policy = -1
        for i in range(1, self.K+1):
            if self.t in self.Tset[i]:
                policy = i
                break
        if policy == -1:
            lassos = []
            for i in range(1, self.K+1):
                if not self.BS["{}_trained".format(i)]: lassos.append(0)
                else: lassos.append(self.BS[i].predict(xt))
            policy = np.argmax(lassos) + 1

        self.l2 = self.l2_orig * np.sqrt((np.log(self.t) + np.log(self.d)) / self.t)
        for i in range(1, self.K+1):
            self.BS[i].set_params(alpha=self.l2)
        
        rt = 0 if policy == yt else -1
        r_t = np.zeros((1, 1))
        r_t[0, 0] = rt
        self.BS[policy].fit(xt, r_t)
        self.BS["{}_trained".format(policy)] = True
        
        return policy

    def test(self, X):
        T, _ = X.shape
        preds = np.zeros((T, self.K))
        for i in range(1, self.K+1):
            preds[:, i-1] = 0 if not self.BS["{}_trained".format(i)] else self.BS[i].predict(X)
        preds = np.argmax(preds, axis=1)
        return preds