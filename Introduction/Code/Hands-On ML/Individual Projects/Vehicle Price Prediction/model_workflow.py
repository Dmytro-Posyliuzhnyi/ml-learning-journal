from pathlib import Path
import pandas as pd
import kagglehub

pd.set_option('display.float_format', '{:,.0f}'.format)


def download_dataset():
    path = kagglehub.dataset_download("nehalbirla/vehicle-dataset-from-cardekho", path='car details v4.csv')
    return pd.read_csv(path)


cars = download_dataset()

print(cars.info())
print(cars.describe())

