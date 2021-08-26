import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer


def identify_merger():
    """This function identifies merged/nonmerged hospitals from verified data,
     and return a dataframe with idenfication of mergers.
    Output: verified data with a new column 'merged' (0: non-merged; 1: merged)"""

    # Upload verified data from Lina as a dataframe
    verified_data = pd.read_csv('closures_verified_2019_04_short.csv')
    # Fill NA in the column 'Type of closure' with 0 (unmerged)
    verified_data['Type of closure'].fillna(0, inplace=True)
    # Define the hospital id as 'merged' if its 'Type of closure' is 2,3,4, otherwise 'unmerged'
    # Find the row whose value of the column 'Type of closure' is 2,3,4
    verified_merged = verified_data.loc[verified_data['Type of closure'].isin([
        2, 3, 4])]

    # Add a new binary column 'merged'
    # First set all 'merged' value into 0
    verified_data['merged'] = 0
    # Replace 0 with 1 for the rows whose original value of the column 'Type of closure' is 2,3,4
    verified_data.loc[verified_merged.index,
                      'merged'] = 1
    print(verified_data)
    return verified_data


def clean_hcris_before_2010(data):
    """This function cleans the HCRIS data before 2010.
    Input: HCRIS data
    Output: HCRIS data after cleaning"""
    # Match the variable names with hcris_2012 and clean the duplicates
    data.rename(columns={'provider': 'id'}, inplace=True)
    data[data["id"].duplicated(keep="last") == True]
    data.drop_duplicates(subset=['id'], keep='last', inplace=True)
    # HCRIS data before 2010 don't have the column "medicaid_days"
    data['medicaid_days'] = np.nan
    data.sort_values(by='id', ascending=True, inplace=True)
    # reset index after cleaning the data
    data.reset_index(drop=True, inplace=True)
    return data


def clean_hcris_after_2010(data):
    """This function cleans the HCRIS data after 2010.
    Input: HCRIS data
    Output: HCRIS data after cleaning"""
    data.rename(columns={'provider': 'id'}, inplace=True)
    data[data["id"].duplicated(keep="last") == True]
    data.drop_duplicates(subset=['id'], keep='last', inplace=True)
    data.sort_values(by='id', ascending=True, inplace=True)
    # reset index after cleaning the data
    data.reset_index(drop=True, inplace=True)
    return data


def clean_hcris_after_2012(data):
    """This function cleans the HCRIS data after 2012.
    Input: HCRIS data
    Output: HCRIS data after cleaning"""
    data.rename(columns={'provider': 'id'}, inplace=True)
    data[data["id"].duplicated(keep="last") == True]
    data['state'] = data['state'].astype(str)
    data['status_cat'] = LabelEncoder().fit_transform(data['status'])
    data['state_cat'] = LabelEncoder().fit_transform(data['state'])
    data['city_cat'] = LabelEncoder().fit_transform(data['city'])
    data['county_cat'] = LabelEncoder().fit_transform(data['county'])
    data = data.drop(['status', 'state', 'city', 'county', 'prvdr_num', 'fyb', 'fybstr', 'fye',
                     'fyestr', 'hospital_name', 'street_addr', 'zip_code', 'medicaid_hmo_discharges'], axis=1)
    data.drop_duplicates(subset=['id'], keep='last', inplace=True)
    # Drop columns with >60% nan
    for column in data:
        count_nan = data[column].isna().sum()
        nan_pct = count_nan/len(data[column])*100
        if nan_pct > 60:
            data = data.drop(column, axis=1)
    data.sort_values(by='id', ascending=True, inplace=True)
    # reset index after cleaning the data
    data.reset_index(drop=True, inplace=True)
    return data


def imputation_method(method, X):
    """This function provides different imputation methods to choose.
    Input: imputation method (i.e., mean, median, knn and multivariate), predictors
    Output: Predictors after imputation"""
    if method == 'knn':
        imp = KNNImputer(n_neighbors=3, weights="uniform")
        X_new = imp.fit_transform(X)
    elif method == 'median':
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
    elif method == 'mean':
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    elif method == 'multivariate':
        imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(X)
    X_new = imp.transform(X)
    return X_new, imp


def fill_col_with_random(df1, column):
    """This function fills df1's column with name 'column' with random data based on non-NaN data from 'column'.
    Input: a dataframe
    Output: the dataframe after random imputation"""
    df2 = df1.copy()
    df2[column] = df2[column].apply(lambda x: np.random.choice(
        df2[column].dropna().values) if np.isnan(x) else x)
    return df2


def resampling(model, X_train, y_train):
    """This function provides resampled data for approach(1).
    Input: resampling method(e.g., SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN), predictors, outcome variables
    Output: Resampled predictors and outcome variables"""
    X_resampled, y_resampled = model.fit_resample(X_train, y_train)
    print(f"New class distribution is: {sorted(Counter(y_resampled).items())}")
    return X_resampled, y_resampled
