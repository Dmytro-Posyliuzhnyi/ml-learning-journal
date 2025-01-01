from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
import tarfile
import urllib.request
import numpy as np

IMAGES_PATH = Path() / "images" / "end_to_end_project"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


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


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def visualize_dataset():
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    housing.hist(bins=50, figsize=(12, 8))
    save_fig("attribute_histogram_plots")
    plt.show()

def visualize_feature_transformation(data, feature, transformed_data):
    # How to use the function
    # transformed_population = np.log(housing['population'])
    # visualize_feature_transformation(housing, 'population', transformed_population)

    plt.figure(figsize=(12, 6))

    # Plot original feature
    plt.subplot(1, 2, 1)
    plt.hist(data[feature], bins=50, color='blue', alpha=0.7)
    plt.title(f"Original '{feature}' Distribution")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    # Plot transformed feature
    plt.subplot(1, 2, 2)
    plt.hist(transformed_data, bins=50, color='green', alpha=0.7)
    plt.title(f"Transformed '{feature}' Distribution")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def visualize_single_feature(data, feature):
    plt.figure(figsize=(12, 6))

    # Plot original feature
    plt.hist(data[feature], bins=50, color='blue', alpha=0.7)
    plt.title(f"'{feature}' Distribution")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def visualize_gaussian_rbf_feature():
    ages = np.linspace(housing["housing_median_age"].min(),
                       housing["housing_median_age"].max(),
                       500).reshape(-1, 1)
    gamma1 = 0.1
    gamma2 = 0.03
    rbf1 = rbf_kernel(ages, [[35]], gamma=gamma1)
    rbf2 = rbf_kernel(ages, [[35]], gamma=gamma2)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Housing median age")
    ax1.set_ylabel("Number of districts")
    ax1.hist(housing["housing_median_age"], bins=50)

    ax2 = ax1.twinx()  # create a twin axis that shares the same x-axis
    color = "blue"
    ax2.plot(ages, rbf1, color=color, label="gamma = 0.10")
    ax2.plot(ages, rbf2, color=color, label="gamma = 0.03", linestyle="--")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel("Age similarity", color=color)

    plt.legend(loc="upper left")
    save_fig("age_similarity_plot")
    plt.show()


def create_income_category_column():
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=['0-1.5', '1.5-3', '3-4.5', '4.5-6', '6-infinity'])


def plot_income_category_histogram(dataset):
    dataset["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
    plt.xlabel("Income category")
    plt.ylabel("Number of districts")
    save_fig("housing_income_cat_bar_plot")  # extra code
    plt.show()


create_income_category_column()
strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"],
                                                   random_state=42)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()


def scatterplot_housing_prices():
    filename = "california.png"
    if not (IMAGES_PATH / filename).is_file():
        homl3_root = "https://github.com/ageron/handson-ml3/raw/main/"
        url = homl3_root + "images/end_to_end_project/" + filename
        print("Downloading", filename)
        urllib.request.urlretrieve(url, IMAGES_PATH / filename)

    housing_renamed = housing.rename(columns={
        "latitude": "Latitude", "longitude": "Longitude",
        "population": "Population",
        "median_house_value": "Median house value (ᴜsᴅ)"})
    housing_renamed.plot(
        kind="scatter", x="Longitude", y="Latitude",
        s=housing_renamed["Population"] / 100, label="Population",
        c="Median house value (ᴜsᴅ)", cmap="jet", colorbar=True,
        legend=True, sharex=False, figsize=(10, 7))

    california_img = plt.imread(IMAGES_PATH / filename)
    axis = -124.55, -113.95, 32.45, 42.05
    plt.axis(axis)
    plt.imshow(california_img, extent=axis)

    save_fig("california_housing_prices_plot")
    plt.show()


def find_correlation(column_name):
    corr_matrix = housing.corr(numeric_only=True)
    print(corr_matrix[column_name].sort_values(ascending=False))


def scatter_matrix_plot():
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    save_fig("scatter_matrix_plot")
    plt.show()


def plot_specific_correlation_scatter_matrix(category1, category2):
    housing.plot(kind="scatter", x=category1, y=category2,
                 alpha=0.1, grid=True)
    save_fig(category1 + "_vs_" + category2)
    plt.show()


def create_additional_attributes():
    housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["people_per_house"] = housing["population"] / housing["households"]


# Remove labels from training set
housing = strat_train_set.drop("median_house_value", axis=1)

# Create a separate set with labels
housing_labels = strat_train_set["median_house_value"].copy()

# Fill in the missing values with median of each attribute
imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
X = imputer.transform(housing_num)

# Transform ocean_proximity into numerical values
housing_cat = housing[["ocean_proximity"]]
one_hot_encoder = OneHotEncoder()
housing_cat_encoded = one_hot_encoder.fit_transform(housing_cat)

# During the next study session I will need to dig more into feature scaling and transformation

# Normalization
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# Standardization
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# Later on I will need to move the functions responsible for visualization to the separate file. 
# This file should be renamed and be responsible for combination and orchestration of different components of the learning process.
