import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings
from pandas.plotting import scatter_matrix
from sklearn import metrics
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import seaborn as sns
from imblearn.pipeline import Pipeline
from skopt import BayesSearchCV


def identify_merger():
    """This function identifies merged/nonmerged hospitals from verified data,
     and return a dataframe with idenfication of mergers.
    Output: verified data with a column 'merged' (0: non-merged; 1: merged)"""

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