import pandas as pd
import numpy as np

# creates np array with class labels according to the doses
# Low = 1, med = 2, high = 3
def convert_to_classes(doses):
    low = 21
    med = 49

    lows = np.where(doses < low, 1, 0)
    meds = np.where(np.all([doses <= med, doses >= low], axis=0), 2, 0)
    highs = np.where(doses > med, 3, 0)

    classes = lows + meds + highs
    return classes

# gives np objects of true doses and true class labels and dataframe of features
def get_data(remove_na=True):
    dataframe = pd.read_csv("data/warfarin.csv")
    dataframe = dataframe[dataframe['Therapeutic Dose of Warfarin'].isna()==False]

    if remove_na:
        dataframe = dataframe[dataframe['Age'].isna()==False]

    doses = dataframe['Therapeutic Dose of Warfarin'].values
    features = dataframe.drop('Therapeutic Dose of Warfarin', axis=1)

    class_labels = convert_to_classes(doses)

    return doses, class_labels, features




# predicts everyone is class 2
def fixed_dose(features):
    preds = np.ones(features.values.shape[0]) + 1

    return preds


def age_to_decade(ages):
    return np.array([int(age[0]) for age in ages])


# predicts based on the clinical dosing algorith (S1f)
# removes labels where height or weight are na
def clinical_alg(features, labels):

    mask = features['Height (cm)'].isna()==False
    labels = labels[mask]
    features = features[mask]

    mask = features['Weight (kg)'].isna()==False
    labels = labels[mask]
    features = features[mask]

    ages = features['Age'].values
    ages = age_to_decade(ages)

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

    # print([v for v in weights if np.isnan(v)])

    return convert_to_classes(dose), labels


# returns the percentage of predictions we got correct
def evaluate_preds(preds, labels):
    total = preds.shape[0]

    num_correct = np.sum(preds == labels)
    return num_correct / total


if __name__ == "__main__":

    doses, class_labels, features  = get_data()

    fixed_preds, labels = fixed_dose(features, class_labels)
    print("fixed dose accuracy: ", evaluate_preds(fixed_preds, labels))

    clinical_preds, labels = clinical_alg(features, class_labels)
    print("clinical dose accuracy: ", evaluate_preds(clinical_preds, labels))












