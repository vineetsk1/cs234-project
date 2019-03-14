import pandas as pd
import numpy as np

# Creates np array with class labels according to the doses.
# Low = 1, med = 2, high = 3
def convert_to_classes(doses):
    low = 21
    med = 49

    lows = np.where(doses < low, 1, 0)
    meds = np.where(np.all([doses <= med, doses >= low], axis=0), 2, 0)
    highs = np.where(doses > med, 3, 0)

    classes = lows + meds + highs
    return classes

# Converts a given decade index to the string representation.
# Eg: 1 would convert to "10 - 19"
def age_to_bin(decade):
    return "%d - %d".format(decade*10, decade*10 + 9)

# Converts decade strings to the index of the bin.
# Eg: ["10 - 19"] would convert to [1]
def ages_to_decades(ages):
    return np.array([int(age[0]) for age in ages])

# Convert arguments array to string (for logging).
def args_to_str(args):
    args = vars(args)
    name = []
    for key in args:
        val = args[key]
        if type(val) != list:
            name.append("{}_{}".format(key, val))
        else:
            name.append("{}_{}".format(key, "_".join(val)))

    return "_".join(name) + ".txt"