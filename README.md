# PHAS0077-project
# Background
This repository is for my project "using machine learning to predict hospital mergers". This project applied seven machine learning algorithms, namely Logistic Regression, Gaussian Naïve Bayes, Linear Discriminant Analysis, Random Forest, Decision Tree, XGBoost and K-Nearest Neighbor. Except XGBoost (**xgboost** package created by Tianqi Chen), all methods used **sklearn** package. Different sampling methods were used to find the best-performing one.

# Installation
**xgboost** and **imblearn** package was required to install for this project.

**imblearn** package was used for undersampling and oversampling methods.

**Numpy**, **pandas**, **seaborn** and **scikit-learn**  package are included in Anaconda and usually don't need to be reinstalled. If not included, install it with pip (pip install scikit-learn).

## Install using `pip` (recommend):
Open your terminal such as git bash, browse to the directory where this file lives, and run
```
$ pip install xgboost
$ pip install imblearn
```
## Install using `conda`:
```
conda install -c conda-forge xgboost
conda install -c anaconda py-xgboost
```
```
conda install -c conda-forge imbalanced-learn
```

# Structure of the repostiory:
* Readme
* gitignore
* Data files (csv files)\
Files named beginning with 'ime_gme' are HCRIS data; 'closures_verified_2019_04_short.csv' is the verified data from Professor Lina Song; 'hosp_closest_info_v5.csv' is the multi-source data; 'verified_hcris_data_2012' is the combined data (verified and HCRIS) I created for checking the study sample.
* data_processing.py\
data_processing.py contains functions used for data processing.
* grid_search.py\
grid_search.py contains functions using GridSearchCV and pipeline to tune hyperparameters for different sampling methods.
* Jupyternotebook files (ipynb files)\
'Using HCRIS (approach 1).ipynb' and 'Using HCRIS (approach 2)-mean/median imputaion.ipynb' applied ML algorithms on HCRIS data using 2 different ways to resample the imbalance data respectively (see section 3.4 in my project). According to the results, approach 2 is the appropriate way to apply resampling methods (section 4.2.2). For 'Using HCRIS (approach 2)-mean imputation' and 'Using HCRIS (approach 2)-median imputation', you could just run the mean one and change the cell under the markdown "impute missing values:" in the notebook to other methods (median, knn and multivariate) using 'imputation_method'. If you want to try random imputation, uncomment the cell below "If using random imputation..." and comment the cell under "Imputate missing values".
'Using multi-source data.ipynb' applied ML algorithms on the multi-source data using approach 2. You could run these ipynb files for results.