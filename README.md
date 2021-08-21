# PHAS0077-project
# Background
This repository is for my project "using machine learning to predict hospital mergers". This project applied seven machine learning algorithms, namely Logistic Regression, Gaussian Naïve Bayes, Linear Discriminant Analysis, Random Forest, Decision Tree, XGBoost and K-Nearest Neighbor. Except XGBoost, all methods used **sklearn** package. Different sampling methods were used to find the best-performing one.

# Installation
**xgboost** and **imblearn** package was required to install for this project.

**imblearn** package was used for undersampling and oversampling methods.


## Install using `pip`:

```
$ pip install xgboost
$ pip install imblearn
```

## Structure of repostiory:
* Readme
* gitignore
* Data files (csv files)\
Files named beginning with 'ime_gme' are HCRIS data; 'closures_verified_2019_04_short.csv' is the verified data from Professor Lina Song; 'hosp_closest_info_v5.csv' is the multi-source data; 'verified_hcris_data_2012' is the combined data (verified and HCRIS) I created for checking the study sample.
* data_processing.py\
data_processing.py contains functions used for data processing.
* grid_search.py\
grid_search.py contains functions using GridSearchCV and pipeline to tune hyperparameters for different sampling methods.