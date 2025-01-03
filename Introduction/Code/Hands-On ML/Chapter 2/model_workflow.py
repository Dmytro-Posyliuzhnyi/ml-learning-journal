from pathlib import Path
from sklearn.model_selection import train_test_split
from pipeline_builder import get_preprocessing_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
import pandas as pd
import tarfile
import urllib.request
import numpy as np


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


housing = load_housing_data()


def create_income_category_column():
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=['0-1.5', '1.5-3', '3-4.5', '4.5-6', '6-infinity'])


def train_model_rscv(pipeline, data):
    param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                      'random_forest__max_features': randint(low=2, high=20)}
    rnd_search = RandomizedSearchCV(
        pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
        scoring='neg_root_mean_squared_error', random_state=42)
    rnd_search.fit(data, housing_labels)
    return rnd_search.best_estimator_


create_income_category_column()
strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"],
                                                   random_state=42)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Remove labels from training set
housing = strat_train_set.drop("median_house_value", axis=1)

# Create a separate set with labels
housing_labels = strat_train_set["median_house_value"].copy()

preprocessing_pipeline = get_preprocessing_pipeline()
housing_prepared = preprocessing_pipeline.fit_transform(housing)

full_pipeline = Pipeline([
    ("preprocessing", preprocessing_pipeline),
    ("random_forest", RandomForestRegressor(random_state=42)),
])

best_model = train_model_rscv(full_pipeline, housing)

# Training RMSE with the best model is around 15k
# Validation RMSE with the best model is around 41k

# I've finally completed Chapter 2. Unfortunately, the RMSE on the training data is much better than on the validation set. 
# The results are very similar to those in the book, but since the book doesn't offer an immediate solution, I'll continue learning, 
# try implementing some ideas on my own, and perhaps revisit this model later to better understand and improve its results.
