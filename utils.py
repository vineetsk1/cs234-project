import pandas as pd
import numpy as np

# creates np array with class labels according to the ground
# truth doses. Low = 1, med = 2, high = 3
def convert_to_classes(doses):
    low = 21
    med = 49

    lows = np.where(doses < low, 1, 0)
    meds = np.where(np.all([doses <= med, doses >= low], axis=0), 2, 0)
    highs = np.where(doses > med, 3, 0)

    classes = lows + meds + highs
    return classes

# gives np objects of true doses, true class labels, features, and names for the features
def get_data():
    dataframe = pd.read_csv("data/warfarin.csv")
    dataframe = dataframe[dataframe['Therapeutic Dose of Warfarin'].isna()==False]

    doses = dataframe['Therapeutic Dose of Warfarin'].values
    features = dataframe.drop('Therapeutic Dose of Warfarin', axis=1)

    class_labels = convert_to_classes(doses)

    return doses, class_labels, features.values, features.columns




# predicts everyone is class 2
def fixed_dose(features):
    preds = np.ones(features.shape[0]) + 1

    return preds

# returns the percentage of predictions we got correct
def evaluate_preds(preds, labels):
    total = preds.shape[0]

    num_correct = np.sum(preds == labels)
    return num_correct / total


if __name__ == "__main__":

    doses, class_labels, features, col_names  = get_data()

    fixed_preds = fixed_dose(features)

    print("fixed dose accuracy: ", evaluate_preds(fixed_preds, class_labels))















