from baseline import FixedBaseline
from baseline import ClinicalBaseline
from linear import LinearUCB

MODEL_DICT = {
    "FixedBaseline": FixedBaseline,
    "ClinicalBaseline": ClinicalBaseline,
    "LinearUCB": LinearUCB
}

def load_model(name, args):
    return MODEL_DICT[name](name, args)