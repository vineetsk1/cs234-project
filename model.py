from baseline import FixedBaseline
from baseline import ClinicalBaseline

MODEL_DICT = {
    "FixedBaseline": FixedBaseline,
    "ClinicalBaseline": ClinicalBaseline
}

def load_model(name, args):
    return MODEL_DICT[name](name, args)