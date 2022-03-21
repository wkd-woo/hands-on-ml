import sys

assert sys.version_info >= (3, 5)
import sklearn

assert sklearn.__version__ >= "0.20"
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = ""
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")
import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "http://leejs.dothome.co.kr/datasets/housing/housing.tgz"
HOUSING_PATH = os.path.join("../", "datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()


fetch_housing_data()

import pandas as pd


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing = strat_train_set.copy()

print("제 03강 실습과제 ####2017265104 장재영\n")

housing = strat_train_set.drop("median_house_value", axis=1)  # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()

median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True)  # option 3
print("\n\n####2017265104 장재영   sample_incomplete_rows (option 3): \n", sample_incomplete_rows)
print("\n\nsample_incomplete_rows (option 3): \n", sample_incomplete_rows["total_bedrooms"])

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)

print("\n\n####2017265104 장재영   imputer.fit(housing_num): \n", imputer.fit(housing_num))
print("\n\n####2017265104 장재영   imputer.statistics_ : \n", imputer.statistics_)
print("\n\n####2017265104 장재영   housing_num.median().values: \n", housing_num.median().values)

X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=list(housing.index.values))
print("\n\n####2017265104 장재영    housing_tr.loc[sample_incomplete_rows.index.values]: \n",
      housing_tr.loc[sample_incomplete_rows.index.values])

print("\n\n\n====================================================================\n")

housing_cat = housing[["ocean_proximity"]]
print(housing_cat)
print("\n####2017265104 장재영    housing_cat.head(10): \n", housing_cat.head(10))

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print("\n####2017265104 장재영    housing_cat_encoded[:10]: \n", housing_cat_encoded[:10])

print("\n####2017265104 장재영    ordinal_encoder.categories_: \n", ordinal_encoder.categories_)

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print("\n####2017265104 장재영    housing_cat_1hot: \n", housing_cat_1hot)

print("\n####2017265104 장재영    NumPy matrix:\n", housing_cat_1hot.toarray())

cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

print("\n####2017265104 장재영    housing_cat_1hot: \n", housing_cat_1hot)

print("\n####2017265104 장재영    cat_encoder.categories_: \n", cat_encoder.categories_)

from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attribs = attr_adder.transform(housing.values)

housing_extra_attribs = pd.DataFrame(housing_extra_attribs, columns=list(housing.columns) + ["rooms_per_household",
                                                                                             "population_per_household"])
print("\n\n####2017265104 장재영    housing_extra_attribs.head(): \n", housing_extra_attribs.head())

print("\n\n\n======================================================================================")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")), ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler()), ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
print("\n####2017265104 장재영    housing_num_tr: \n", housing_num_tr)

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs), ])

housing_prepared = full_pipeline.fit_transform(housing)

print("\n####2017265104 장재영    Housing prepared:", housing_prepared)

print("\n####2017265104 장재영    Housing prepared shape:", housing_prepared.shape)

print("\n\n=====================================================================================")

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# let's try the full pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("\n####2017265104 장재영    Predictions:\n", lin_reg.predict(some_data_prepared))
print("\n####2017265104 장재영    Labels:\n", list(some_labels))

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("\n####2017265104 장재영    RMSE:\n", lin_rmse)

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("\nTree RMSE:\n", tree_rmse)

print("\n\n=====================================================================================")

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("\n####2017265104 장재영    Scores:\n", scores)
    print("\n####2017265104 장재영    Mean:\n", scores.mean())
    print("\n####2017265104 장재영    Standard deviation:\n", scores.std())


print("\n\n================Compute the same scores for the Linear Regression================\n")

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
print("\n\n####2017265104 장재영    display_scores(lin_rmse_scores): \n", display_scores(lin_rmse_scores))

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print("\n####2017265104 장재영    RandomForestRegressor RMSE:\n", forest_rmse)

print("\n\n=====================================================================================\n\n")
print("GridSearchCV: \n")

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}, ]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

print("\n####2017265104 장재영    grid_search.fit(housing_prepared, housing_labels) : \n",
      grid_search.fit(housing_prepared, housing_labels))

print("\n####2017265104 장재영    grid_search.best_params_ : \n", grid_search.best_params_)
print("\n####2017265104 장재영    grid_search.best_estimator_ : \n", grid_search.best_estimator_)
print("\n\n=====================================================================================\n")
print("RandomizedSearchCV: \n")

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'n_estimators': randint(low=1, high=200), 'max_features': randint(low=1, high=8), }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs, n_iter=10, cv=5,
                                scoring='neg_mean_squared_error', random_state=42)

print("\n####2017265104 장재영    rnd_search.fit(housing_prepared, housing_labels) : \n",
      rnd_search.fit(housing_prepared, housing_labels))

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
print("\n####2017265104 장재영    feature_importances: \n", feature_importances)

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print("\n####2017265104 장재영    sorted(zip(feature_importances, attributes), reverse=True) : \n",
      sorted(zip(feature_importances, attributes), reverse=True))

print("\n\n=====================================================================================\n")
print("Evaluate your system on the Test Set\n")

final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)

print("####2017265104 장재영    y_test: \n", y_test[:10])
print("####2017265104 장재영    final_predictions: \n", final_predictions[:10])
print("\n####2017265104 장재영    Final RMSE:\n", final_rmse)
