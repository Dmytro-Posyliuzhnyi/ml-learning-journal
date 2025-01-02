from pathlib import Path
from sklearn.model_selection import train_test_split
from pipeline_builder import get_preprocessing_pipeline
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
