import numpy as np
from utils import ages_to_decades
from utils import convert_to_classes

class Model():
    def __init__(self, name, args, baseline):
        self.name = name
        self.args = args
        self.baseline = baseline

    def initialize(self, features):
        raise Exception("Invalid model {}, please use subclass.".format(name))

    # Training
    # Fits a single example, updates model's parameters
    def train(self, x, y):
        raise Exception("Invalid model {}, please use subclass.".format(name))

    # Testing
    # Evaluates model on given features
    def test(self, features):
        raise Exception("Invalid model {}, please use subclass.".format(name))

class FixedBaseline(Model):
    def __init__(self, name, args):
        super().__init__(name, args, True)

    def initialize(self, features):
        raise Exception("Invalid model {}, baseline does not have parameters.".format(name))

    def train(self, x, y):
        raise Exception("Invalid model {}, baseline does not train.".format(name))

    # Always predict class 2.
    def test(self, features):
        preds = np.ones(features.values.shape[0]) + 1
        return preds

class ClinicalBaseline(Model):
    def __init__(self, name, args):
        super().__init__(name, args, True)

    def initialize(self, features):
        raise Exception("Invalid model {}, baseline does not have parameters.".format(name))

    def train(self, x, y):
        raise Exception("Invalid model {}, baseline does not train.".format(name))

    def test(self, features):
        ages = features['Age'].values
        ages = ages_to_decades(ages)

        heights = features['Height (cm)'].values
        weights = features['Weight (kg)'].values
        asian = features['Race'].values == 'Asian'
        black = features['Race'].values == 'Black or African American'
        missing_race = features['Race'].values == 'Unknown'

        enzyme_inducer = []
        for drug in ["Carbamazepine (Tegretol)", "Phenytoin (Dilantin)", "Rifampin or Rifampicin"]:
            d = features[drug].values == 1
            enzyme_inducer.append(d)
        enzyme_inducer = np.any(enzyme_inducer, axis=0)

        amiodarone = features['Amiodarone (Cordarone)'].values == 1

        dose = ages * -.2546 + .0118 * heights + .0134 * weights
        dose += -.6752 * asian + .4060 * black + .0443 * missing_race
        dose += 1.2799 * enzyme_inducer - .5695 * amiodarone
        dose += 4.0376
        dose = np.square(dose)
        return convert_to_classes(dose)