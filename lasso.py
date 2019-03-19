import numpy as np
import pandas as pd
from baseline import Model

class LassoBandit(Model):
    def __init__(self, name, args):
        super().__init__(name, args, False)

    def initialize(self, features):
    	pass

    def train(self, xt, yt):
    	pass

    def test(self, X):
    	pass