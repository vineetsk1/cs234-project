from baseline import FixedBaseline
from baseline import ClinicalBaseline
from linear import LinearUCB
from lasso import LassoBandit

MODEL_DICT = {
    "FixedBaseline": FixedBaseline,
    "ClinicalBaseline": ClinicalBaseline,
    "LinearUCB": LinearUCB,
    "LassoBandit": LassoBandit
}

def load_model(name, args):
    return MODEL_DICT[name](name, args)