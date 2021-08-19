import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings
from pandas.plotting import scatter_matrix
from sklearn import metrics, pipeline
from sklearn import ensemble
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


def grid_search_lr(sampling, X_test, y_test, X_train, y_train):
    """This function uses grid search to find the hyperparameters with the best f1 score for Logistic regression.
    Output: training f1, test f1, auprc, confusion matrix and the best classifier"""
    lr = LogisticRegression(solver='liblinear')
    pipeline = Pipeline(steps=[['sampling', sampling],
                               ['classifier', lr]])
    param_grid_ = {'C': [
        0.1, 1, 10, 100, 200, 500, 1000], 'penalty': ['l1', 'l2']}
    param_grid_clf = {'classifier__C': [
        0.1, 1, 10, 100, 200, 500, 1000], 'classifier__penalty': ['l1', 'l2']}
    if sampling is None:
        estimator = lr
        param_grid = param_grid_
    else:
        estimator = pipeline
        param_grid = param_grid_clf
    # Fitting grid search to the train data with 5 folds
    gridsearch = GridSearchCV(estimator=estimator,
                              param_grid=param_grid,
                              cv=StratifiedKFold(
                                  n_splits=5, random_state=1, shuffle=True),
                              scoring='f1')
    gridsearch.fit(X_train, y_train)
    y_pred = gridsearch.predict(X_test)
    y_pred_proba = gridsearch.predict_proba(X_test)[:, 1]
    print("Best: %f using %s" %
          (gridsearch.best_score_, gridsearch.best_params_))
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Calculating and printing the f1 score
    f1_train = gridsearch.best_score_
    f1_test = f1_score(y_test, y_pred)
    print('The f1 score for the testing data:', f1_test)
    auprc = average_precision_score(y_test, y_pred_proba)
    return f1_train, f1_test, auprc, conf_matrix, gridsearch


def grid_search_lda(sampling, X_test, y_test, X_train, y_train):
    """This function uses grid search to find the hyperparameters with the best f1 score for LinearDiscriminantAnalysis.
    Output: training f1, test f1, auprc, confusion matrix and the best classifier"""
    lr = LogisticRegression(solver='liblinear')
    lda = LinearDiscriminantAnalysis()
    pipeline = Pipeline(steps=[['sampling', sampling],
                               ['classifier', lda]])
    param_grid_ = {
        'solver': ['svd', 'lsqr', 'eigen'],
        'tol': [1e-05, 0.0001, 0.0003]
    }
    param_grid_clf = {
        'classifier__solver': ['svd', 'lsqr', 'eigen'],
        'classifier__tol': [1e-05, 0.0001, 0.0003]
    }
    if sampling is None:
        estimator = lda
        param_grid = param_grid_
    else:
        estimator = pipeline
        param_grid = param_grid_clf
    # Fitting grid search to the train data with 5 folds
    gridsearch = GridSearchCV(estimator=estimator,
                              param_grid=param_grid,
                              cv=StratifiedKFold(
                                  n_splits=5, random_state=1, shuffle=True),
                              scoring='f1')
    gridsearch.fit(X_train, y_train)
    y_pred = gridsearch.predict(X_test)
    y_pred_proba = gridsearch.predict_proba(X_test)[:, 1]
    print("Best: %f using %s" %
          (gridsearch.best_score_, gridsearch.best_params_))
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Calculating and printing the f1 score
    f1_train = gridsearch.best_score_
    f1_test = f1_score(y_test, y_pred)
    print('The f1 score for the testing data:', f1_test)
    auprc = average_precision_score(y_test, y_pred_proba)
    return f1_train, f1_test, auprc, conf_matrix, gridsearch


def grid_search_knn(sampling, X_test, y_test, X_train, y_train):
    """This function uses grid search to find the hyperparameters with the best f1 score for KNN.
    Output: training f1, test f1, auprc, confusion matrix and the best classifier"""
    knn = KNeighborsClassifier()
    pipeline = Pipeline(steps=[['sampling', sampling],
                               ['classifier', knn]])
    param_grid_ = {'weights': ['uniform', 'distance'], 'n_neighbors': [
        3, 5, 10], 'algorithm': ['auto', 'ball_tree', 'kd_tree'], 'leaf_size': [2, 5, 10, 20, 30]}
    param_grid_clf = {'classifier__weights': ['uniform', 'distance'], 'classifier__n_neighbors': [
        3, 5, 10], 'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree'], 'classifier__leaf_size': [2, 5, 10, 20, 30]}
    if sampling is None:
        estimator = knn
        param_grid = param_grid_
    else:
        estimator = pipeline
        param_grid = param_grid_clf
    # Fitting grid search to the train data with 5 folds
    gridsearch = GridSearchCV(estimator=estimator,
                              param_grid=param_grid,
                              cv=StratifiedKFold(
                                  n_splits=5, random_state=1, shuffle=True),
                              scoring='f1')
    gridsearch.fit(X_train, y_train)
    y_pred = gridsearch.predict(X_test)
    y_pred_proba = gridsearch.predict_proba(X_test)[:, 1]
    print("Best: %f using %s" %
          (gridsearch.best_score_, gridsearch.best_params_))
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Calculating and printing the f1 score
    f1_train = gridsearch.best_score_
    f1_test = f1_score(y_test, y_pred)
    print('The f1 score for the testing data:', f1_test)
    auprc = average_precision_score(y_test, y_pred_proba)
    return f1_train, f1_test, auprc, conf_matrix, gridsearch

def grid_search_dt(sampling, X_test, y_test, X_train, y_train):
    """This function uses grid search to find the hyperparameters with the best f1 score for Decision Tree.
    Output: training f1, test f1, auprc, confusion matrix and the best classifier"""
    dt = DecisionTreeClassifier()
    pipeline = Pipeline(steps=[['sampling', sampling],
                               ['classifier', dt]])
    param_grid_ = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['random', 'best'],
        'max_depth': [1, 2, 10],
        'min_samples_leaf': [1, 2, 10]
    }
    param_grid_clf = {
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__splitter': ['random', 'best'],
        'classifier__max_depth': [1, 2, 10],
        'classifier__min_samples_leaf': [1, 2, 10]
    }
    if sampling is None:
        estimator = dt
        param_grid = param_grid_
    else:
        estimator = pipeline
        param_grid = param_grid_clf
    # Fitting grid search to the train data with 5 folds
    gridsearch = GridSearchCV(estimator=estimator,
                              param_grid=param_grid,
                              cv=StratifiedKFold(
                                  n_splits=5, random_state=1, shuffle=True),
                              scoring='f1')
    gridsearch.fit(X_train, y_train)
    y_pred = gridsearch.predict(X_test)
    y_pred_proba = gridsearch.predict_proba(X_test)[:, 1]
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Best: %f using %s" %
          (gridsearch.best_score_, gridsearch.best_params_))
    # Calculating and printing the f1 score
    f1_train = gridsearch.best_score_
    f1_test = f1_score(y_test, y_pred)
    print('The f1 score for the testing data:', f1_test)
    auprc = average_precision_score(y_test, y_pred_proba)
    return f1_train, f1_test, auprc, conf_matrix, gridsearch

def grid_search_rf(sampling, X_test, y_test, X_train, y_train):
    """This function uses grid search to find the hyperparameters with the best f1 score for Random Forest.
    Output: training f1, test f1, auprc, confusion matrix and the best classifier"""
    rf = RandomForestClassifier()
    pipeline = Pipeline(steps=[['sampling', sampling],
                               ['classifier', rf]])
    param_grid_ = {
        'max_depth': [2, 5, 10, 20],
        'n_estimators': [10, 100, 1000],
    }
    param_grid_clf = {
        'classifier__max_depth': [2, 5, 10, 20],
        'classifier__n_estimators': [10, 100, 1000],
        # 'classifier__criterion': ['gini', 'entropy']
    }
    if sampling is None:
        estimator = rf
        param_grid = param_grid_
    else:
        estimator = pipeline
        param_grid = param_grid_clf
    # Fitting grid search to the train data with 5 folds
    gridsearch = GridSearchCV(estimator=estimator,
                              param_grid=param_grid,
                              cv=StratifiedKFold(
                                  n_splits=5, random_state=1, shuffle=True),
                              scoring='f1')
    gridsearch.fit(X_train, y_train)
    y_pred = gridsearch.predict(X_test)
    y_pred_proba = gridsearch.predict_proba(X_test)[:, 1]
    print("Best: %f using %s" %
          (gridsearch.best_score_, gridsearch.best_params_))
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Calculating and printing the f1 score
    f1_train = gridsearch.best_score_
    f1_test = f1_score(y_test, y_pred)
    print('The f1 score for the testing data:', f1_test)
    auprc = average_precision_score(y_test, y_pred_proba)
    return f1_train, f1_test, auprc, conf_matrix, gridsearch

def grid_search_xgb(sampling, X_test, y_test, X_train, y_train):
    """This function uses grid search to find the hyperparameters with the best f1 score for XGBoost.
    Output: training f1, test f1, auprc, confusion matrix and the best classifier"""
    xgboost = XGBClassifier()
    pipeline = Pipeline(steps=[['sampling', sampling],
                               ['classifier', xgboost]])
    param_grid_ = {
        'n_estimators': [100, 1000, 10000],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    param_grid_clf = {
        'classifier__n_estimators': [100, 1000, 10000],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }
    if sampling is None:
        estimator = xgboost
        param_grid = param_grid_
    else:
        estimator = pipeline
        param_grid = param_grid_clf
    # Fitting grid search to the train data with 5 folds
    gridsearch = GridSearchCV(estimator=estimator,
                              param_grid=param_grid,
                              cv=StratifiedKFold(
                                  n_splits=5, random_state=1, shuffle=True),
                              scoring='f1')
    gridsearch.fit(X_train, y_train)
    y_pred = gridsearch.predict(X_test)
    y_pred_proba = gridsearch.predict_proba(X_test)[:, 1]
    print("Best: %f using %s" %
          (gridsearch.best_score_, gridsearch.best_params_))
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Calculating and printing the f1 score
    f1_train = gridsearch.best_score_
    f1_test = f1_score(y_test, y_pred)
    print('The f1 score for the testing data:', f1_test)
    auprc = average_precision_score(y_test, y_pred_proba)
    return f1_train, f1_test, auprc, conf_matrix, gridsearch