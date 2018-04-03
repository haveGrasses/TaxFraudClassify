"""
interpreter: py35
three methods to fill missing data
"""
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import Imputer
from scipy.interpolate import lagrange


raw_data = pd.read_csv('./input/missing_data.csv', encoding='utf-8', header=None)
# print(raw_data)
# show the sum of missing data number in every column
# print(raw_data.isnull().sum().sort_values(ascending=False))

# the first method: pandas.Series.interpolate
filled_data = DataFrame()
for column in raw_data.columns:
    print(column)
    filled_data[column] = raw_data[column].interpolate(method='pchip', axis=0)  # method: linear, zero, ...
print(filled_data)

# the second method: imputer
imr = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)  # strategy: mean, median, most_frequent
imr = imr.fit(raw_data)
data_filledBy_imr = DataFrame(imr.transform(raw_data))
print(data_filledBy_imr)

# the third method: scipy.interpolate


def interp(series, index, k=5):
    y = series[list(range(index-k, index)) + list(range(index+1, index+1+k))]  # get k*2 numbers to predict
    y = y[y.notnull()]  # remove nan
    return lagrange(y.index, list(y))(index)  # lagrange()返回多项式，(index)代入index到多项式求值


for i in raw_data.columns:
    for j in range(len(raw_data)):
        if (raw_data[i].isnull())[j]:
            raw_data[i][j] = interp(raw_data[i], j)
print(raw_data)


