import pandas as pd
import numpy as np

from utils import convert_to_classes
from copy import deepcopy
from collections import defaultdict
from scipy.stats import percentileofscore

def imputeMeasurements(df_orig):
    df = deepcopy(df_orig)
    race_gend_avg_height = defaultdict(list)
    race_gend_avg_weight = defaultdict(list)
    men_weight = list(df[(df['Gender']=='male')&(df['Weight (kg)'].isna()==False)]['Weight (kg)'])
    women_weight = list(df[(df['Gender']=='female')&(df['Weight (kg)'].isna()==False)]['Weight (kg)'])
    men_height = list(df[(df['Gender']=='male')&(df['Height (cm)'].isna()==False)]['Height (cm)'])
    women_height = list(df[(df['Gender']=='female')&(df['Height (cm)'].isna()==False)]['Height (cm)'])
    for _, row in df.iterrows():
        if pd.isnull(row['Race']) or pd.isnull(row['Gender']):
            continue
        key = row['Race']+row['Gender']
        if not pd.isnull(row['Height (cm)']):
            race_gend_avg_height[key].append(row['Height (cm)'])
        if not pd.isnull(row['Weight (kg)']):
            race_gend_avg_weight[key].append(row['Weight (kg)'])
            
    for i,row in df.iterrows():
        if not (pd.isnull(row['Height (cm)']) or pd.isnull(row['Weight (kg)'])) or pd.isnull(row['Gender']):
            continue
        if pd.isnull(row['Height (cm)']) and pd.isnull(row['Weight (kg)']):
            if pd.isnull(row['Race']):
                if row['Gender'] == 'male':
                    df.at[i, 'Height (cm)'] = np.mean(men_height)
                    df.at[i,'Weight(kg)'] = np.mean(men_weight)
                else:
                    df.at[i, 'Height (cm)'] = np.mean(women_height)
                    df.at[i,'Weight(kg)'] = np.mean(women_weight)
            else:
                key = row['Race']+row['Gender']
                df.at[i, 'Height (cm)'] = np.mean(race_gend_avg_height[key])
                df.at[i,'Weight (kg)'] = np.mean(race_gend_avg_weight[key])
        else:
            if pd.isnull(row['Height (cm)']):
                if pd.isnull(row['Race']):
                    if row['Gender'] == 'male':
                        perc = percentileofscore(men_weight, row['Weight (kg)'])
                        df.at[i,'Height (cm)'] = np.percentile(men_height, perc)
                    else:
                        perc = percentileofscore(women_weight, row['Weight (kg)'])
                        df.at[i,'Height (cm)'] = np.percentile(women_height, perc)
                else:
                    height_arr = race_gend_avg_height[row['Race']+row['Gender']]
                    weight_arr = race_gend_avg_weight[row['Race']+row['Gender']]
                    perc = percentileofscore(weight_arr, row['Weight (kg)'])
                    df.at[i,'Height (cm)'] = np.percentile(height_arr, perc)
            else:
                if pd.isnull(row['Race']):
                    if row['Gender'] == 'male':
                        perc = percentileofscore(men_height, row['Height (cm)'])
                        df.at[i,'Weight (kg)'] = np.percentile(men_weight, perc)
                    else:
                        perc = percentileofscore(women_height, row['Height (cm)'])
                        df.at[i,'Weight (kg)'] = np.percentile(women_weight, perc)
                else:
                    height_arr = race_gend_avg_height[row['Race']+row['Gender']]
                    weight_arr = race_gend_avg_weight[row['Race']+row['Gender']]
                    perc = percentileofscore(height_arr, row['Height (cm)'])
                    df.at[i,'Weight (kg)'] = np.percentile(weight_arr, perc)
            
    return df

# Loads the data, removes rows missing GT data,
# processes missing entries for age/height/weight
# (either by removing or using the average),
# and returns the features, the doses, and the 
# corresponding class labels.
def get_data(args):

    # Load data, remove rows missing the GT data.
    df = pd.read_csv("data/warfarin.csv")
    df = df.dropna(subset=['Therapeutic Dose of Warfarin'])
    df = df.dropna(subset=['Age'])

    if args.impute_type:
        df = imputeMeasurements(df)
    else:
        avg_height = df['Height (cm)'].dropna().mean()
        df['Height (cm)'].fillna(avg_height, inplace=True)
        avg_weight = df['Weight (kg)'].dropna().mean()
        df['Weight (kg)'].fillna(avg_weight, inplace=True)

    avg_inr = df['Target INR'].dropna().mean()
    df['Target INR'].fillna(avg_inr, inplace=True)

    # For these columns, convert from float type to object:
    # These are indicator variables and should be binary classes.
    # After this step, the remaining numeric columns are:
    # Height, Weight, Target INR
    cols = [
        'Diabetes', 'Congestive Heart Failure and/or Cardiomyopathy', 'Valve Replacement',
        'Aspirin', 'Acetaminophen or Paracetamol (Tylenol)', 'Was Dose of Acetaminophen or Paracetamol (Tylenol) >1300mg/day',
        'Simvastatin (Zocor)', 'Atorvastatin (Lipitor)', 'Fluvastatin (Lescol)',
        'Lovastatin (Mevacor)', 'Pravastatin (Pravachol)', 'Rosuvastatin (Crestor)',
        'Cerivastatin (Baycol)', 'Amiodarone (Cordarone)', 'Carbamazepine (Tegretol)',
        'Phenytoin (Dilantin)', 'Rifampin or Rifampicin', 'Sulfonamide Antibiotics',
        'Macrolide Antibiotics', 'Anti-fungal Azoles', 'Herbal Medications, Vitamins, Supplements',
        'Subject Reached Stable Dose of Warfarin', 'Current Smoker']
    for col in cols:
        df[col] = df[col].astype(str)

    doses = df['Therapeutic Dose of Warfarin'].values
    features = df.drop([
        'Therapeutic Dose of Warfarin', 'PharmGKB Subject ID',
        'INR on Reported Therapeutic Dose of Warfarin',
        'Unnamed: 63', 'Unnamed: 64', 'Unnamed: 65'], axis=1)

    labels = convert_to_classes(doses)

    return features, doses, labels