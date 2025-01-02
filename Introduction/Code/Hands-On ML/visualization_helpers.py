import matplotlib.pyplot as plt
from pathlib import Path
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import urllib.request


IMAGES_PATH = Path() / "images" / "end_to_end_project"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def visualize_dataset(dataset):
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    dataset.hist(bins=50, figsize=(12, 8))
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


def visualize_gaussian_rbf_feature(dataset):
    ages = np.linspace(dataset["housing_median_age"].min(),
                       dataset["housing_median_age"].max(),
                       500).reshape(-1, 1)
    gamma1 = 0.1
    gamma2 = 0.03
    rbf1 = rbf_kernel(ages, [[35]], gamma=gamma1)
    rbf2 = rbf_kernel(ages, [[35]], gamma=gamma2)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Housing median age")
    ax1.set_ylabel("Number of districts")
    ax1.hist(dataset["housing_median_age"], bins=50)

    ax2 = ax1.twinx()  # create a twin axis that shares the same x-axis
    color = "blue"
    ax2.plot(ages, rbf1, color=color, label="gamma = 0.10")
    ax2.plot(ages, rbf2, color=color, label="gamma = 0.03", linestyle="--")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel("Age similarity", color=color)

    plt.legend(loc="upper left")
    save_fig("age_similarity_plot")
    plt.show()


def plot_specific_correlation_scatter_matrix(dataset, category1, category2):
    dataset.plot(kind="scatter", x=category1, y=category2,
                 alpha=0.1, grid=True)
    save_fig(category1 + "_vs_" + category2)
    plt.show()


def plot_income_category_histogram(dataset):
    dataset["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
    plt.xlabel("Income category")
    plt.ylabel("Number of districts")
    save_fig("housing_income_cat_bar_plot")  # extra code
    plt.show()


def scatterplot_housing_prices(dataset):
    filename = "california.png"
    if not (IMAGES_PATH / filename).is_file():
        homl3_root = "https://github.com/ageron/handson-ml3/raw/main/"
        url = homl3_root + "images/end_to_end_project/" + filename
        print("Downloading", filename)
        urllib.request.urlretrieve(url, IMAGES_PATH / filename)

    housing_renamed = dataset.rename(columns={
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


def find_correlation(dataset, column_name):
    corr_matrix = dataset.corr(numeric_only=True)
    print(corr_matrix[column_name].sort_values(ascending=False))


def scatter_matrix_plot(dataset):
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(dataset[attributes], figsize=(12, 8))
    save_fig("scatter_matrix_plot")
    plt.show()
