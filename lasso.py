import numpy as np
import pandas as pd
from baseline import Model

from sklearn.linear_model import Lasso

class LassoBandit(Model):
    def __init__(self, name, args):
        super().__init__(name, args, False)

    def initialize(self, features):
        self.q = self.args.q
        self.k = 3
        self.T, self.d = features.shape
        self.l2 = self.args.lambda2

        self.TS, self.BS = {}, {}
        self.Sinit = {}
        for i in range(1, self.k+1):
            self.TS[i] = set()
            self.BS[i] = Lasso(alpha=self.l2, warm_start=True, max_iter=1000)
            self.Sinit[i] = False

        for i in range(1, self.k+1):
            nextarm = False
            for n in range(0, 1000):
                for j in range(self.q*(i-1)+1, self.q*i+1):
                    t = (pow(2, n) - 1) * self.k * self.q + j
                    if t > self.T:
                        nextarm = True
                        break
                    self.TS[i].add(t)
                if nextarm:
                    break

        self.t = 0

    def train(self, xt, yt):
        self.t += 1
        policy = -1
        xt = xt.reshape((1, xt.shape[0])) # Shape (1, d)
        for i in range(1, self.k+1):
            if self.t in self.TS[i]:
                policy = i
                break
        if policy == -1:
            Ks = []
            for i in range(1, self.k+1):
                if not self.Sinit[i]:
                    Ks.append(0)
                else:
                    Ks.append(self.BS[i].predict(xt))
            print(Ks)
            policy = np.argmax(Ks) + 1

        self.l2 = self.args.lambda2 * np.sqrt((np.log(self.t) + np.log(self.d))/self.t)
        reward = np.zeros((1,))
        reward[0] = 0 if yt == policy else -1

        # print("T", self.t, "Y", yt, "Policy", policy, "Reward", reward.item())
        self.Sinit[policy] = True 
        self.BS[policy].fit(xt, reward)
        return policy


    def test(self, X):
        T, _ = X.shape
        preds = np.zeros((T, self.k))
        for i in range(1, self.k+1):
            preds[:, i-1] = 0 if not self.Sinit[i] else self.BS[i].predict(X)
        preds = np.argmax(preds, axis=1)
        return preds