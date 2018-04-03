"""
interpreter: py35
three methods to fill missing data
"""
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import Imputer
from scipy.interpolate import lagrange


raw_data = pd.read_csv('./input/missing_data.csv', encoding='utf-8', header=None)
# print(raw_data)
# show the sum of missing data number in every column
# print(raw_data.isnull().sum().sort_values(ascending=False))

# the first method: interpolate
filled_data = DataFrame()
for column in raw_data.columns:
    print(column)
    filled_data[column] = raw_data[column].interpolate(method='pchip', axis=0)  # method: linear, zero, ...
print(filled_data)

# the second method: imputer
imr = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)  # strategy: mean, median, most_frequent
imr = imr.fit(raw_data)
data_filledBy_imr = imr.transform(raw_data)