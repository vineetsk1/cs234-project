import pandas as pd
import numpy as np

from utils import convert_to_classes

# Loads the data, removes rows missing GT data,
# processes missing entries for age/height/weight
# (either by removing or using the average),
# and returns the features, the doses, and the 
# corresponding class labels.
def get_data(drop_age, drop_height, drop_weight):

    # Load data, remove rows missing the GT data.
    df = pd.read_csv("data/warfarin.csv")
    df = df.dropna(subset=['Therapeutic Dose of Warfarin'])

    if drop_age:
        df = df.dropna(subset=['Age'])
    else:
        decades = [int(age[0]) for age in df['Age'].dropna()]
        avg_age = age_to_bin(int(np.mean(decades)))
        df['Age'].fillna(avg_age, inplace=True)

    if drop_height:
        df = df.dropna(subset=['Height (cm)'])
    else:
        avg_height = df['Height (cm)'].dropna().mean()
        df['Height (cm)'].fillna(avg_height, inplace=True)

    if drop_weight:
        df = df.dropna(subset=['Weight (kg)'])
    else:
        avg_weight = df['Weight (kg)'].dropna().mean()
        df['Weight (kg)'].fillna(avg_weight, inplace=True)

    doses = df['Therapeutic Dose of Warfarin'].values
    features = df.drop('Therapeutic Dose of Warfarin', axis=1)
    labels = convert_to_classes(doses)

    return features, doses, labels