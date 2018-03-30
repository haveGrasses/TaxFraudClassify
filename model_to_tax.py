# -*- coding: utf-8 -*-
"""
interpreter:py 3.5
"""
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn import metrics
from keras import Sequential
from keras.layers.core import Dense, Activation
from sklearn.tree import DecisionTreeClassifier
plt.rcParams['font.sans-serif'] = ['SimHei']
df_raw = pd.read_csv('./input/汽车销售纳税人偷漏税文件.csv', encoding='utf-8', index_col=0)
# print(df_raw.head(), df_raw.shape, df_raw.describe())

# view the output
# df_raw["输出"].value_counts().plot('bar')
# plt.show()
# 正负样本基本均衡

# check if missing value exists
# print(df_raw.isnull().sum().sort_values(ascending=False))

# exploring the relationship between every x and y


def plot_relation(x, xname):
    """
    plot a pic to show how the value of x affect y
    :param x: a series like: raw_df['***']
    :param xname: pic title
    :return:
    """
    y = df_raw['输出']
    group = y.groupby(x).count()
    group.plot(kind='bar', alpha=0.5)
    plt.title(xname)
    plt.show()


# for i in range(len(df_raw.columns)-1):
#     columns = df_raw.columns.tolist()
#     x = df_raw['%s' % columns[i]]
#     plot_relation(x, columns[i])
# as the plot pic indicates, all x will affect is_fraud, so do not need to choose related parameters, use all

# reformat the raw_df
# print(df_raw.dtypes)
# change the value of '输出' to 0 or 1


def encode_y(row):
    if row['输出'] == '正常':
        return 0
    else:
        return 1


df_raw['输出'] = df_raw.apply(encode_y, axis=1)
# print(df_raw['输出'].head())
# rename columns
df_raw.rename(columns={'销售类型': 'sales_type',
                       '销售模式': 'sales_mode',
                       '汽车销售平均毛利': 'margin_of_sales',
                       '维修毛利': 'margin_of_repair',
                       '企业维修收入占销售收入比重': 'repair_sales_ratio',
                       '增值税税负': 'value-added_tax_burden',
                       '存货周转率': 'inventory_turnover',
                       '成本费用利润率': 'profits_to_costs',
                       '整体理论税负': 'overall_tax_burden',
                       '整体税负控制数': 'overall_tax_burden_limit',
                       '办牌率': 'license_rates',
                       '单台办牌手续费收入': 'income_of_license',
                       '代办保险率': 'insurance_rates',
                       '保费返还率': 'premium_return_rates',
                       '输出': 'is_fraud'
                       }, inplace=True)
# one-hot encoding
df_dummy = pd.get_dummies(df_raw)
print(df_dummy.head())
# after one-hot, the df was changed from 124*15 to 124*26

# pre-process numerical columns: standardize the data: [x-mean(x)]/std
numeric_cols = df_raw.columns[df_raw.dtypes == 'float64']
# print(numeric_cols)
numeric_cols_means = df_dummy.loc[:, numeric_cols].mean()
numeric_cols_std = df_dummy.loc[:, numeric_cols].std()
df_dummy.loc[:, numeric_cols] = (df_dummy.loc[:, numeric_cols] - numeric_cols_means) / numeric_cols_std
df_dummy_x = df_dummy.drop('is_fraud', 1)
# split data
X_train, X_test, y_train, y_test = train_test_split(df_dummy_x, df_dummy['is_fraud'],
                                                    test_size=0.2, random_state=0)
print(X_train.head(), y_test.head(), y_train.head())

# LR regression
lr = LR()

# bagging
params = [1, 10, 15, 20, 25, 30, 40]
test_scores2 = []
for param in params:
    clf = BaggingClassifier(n_estimators=param)
    test_score2 = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores2.append(np.mean(test_score2))
bg_best_param = params[test_scores2.index(min(test_scores2))]
plt.plot(params, test_scores2)
plt.title("bagging: n_estimator vs CV Error")
plt.show()
# when param=25, the cv error hits the bottom. the minimum error:0.315, smaller the only use ridge

# xgboost
params = [1, 2, 3, 4, 5, 6]
test_scores3 = []
for param in params:
    clf = XGBClassifier(max_depth=param)
    test_score3 = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores3.append(np.mean(test_score3))
xgb_best_param = params[test_scores3.index(min(test_scores3))]
plt.plot(params, test_scores3)
plt.title("xgb: max_depth vs CV Error")
plt.show()
# xgboost 1棵树就能达到0.31的小误差。。。

# random forest
max_features = [.1, .3, .5, .7, .9, .99]
test_scores4 = []
for max_feat in max_features:
    clf = RandomForestClassifier(n_estimators=100, max_features=max_feat)
    test_score4 = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores4.append(np.mean(test_score4))
rf_best_param = params[test_scores4.index(min(test_scores4))]
plt.plot(max_features, test_scores4)
plt.title("rf: Max Features vs CV Error")
plt.show()
# max_feature=0.7, cv error:0.329, 没有xgboost好


bg = BaggingClassifier(n_estimators=bg_best_param)
rf = RandomForestClassifier(n_estimators=100, max_features=rf_best_param)
xgb = XGBClassifier(max_depth=xgb_best_param)

lr.fit(X_train, y_train)
bg.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)


y_rf = rf.predict(X_test)
y_bg = bg.predict(X_test)
y_lr = lr.predict(X_test)
y_xgb = xgb.predict(X_test)

print(y_lr, y_rf, y_xgb)


def print_metrics(true_values, predicted_values):
    print("Accuracy:\n", metrics.accuracy_score(true_values, predicted_values), '\n',
          "AUC:\n", metrics.roc_auc_score(true_values, predicted_values), '\n',
          "Confusion Matrix:\n", metrics.confusion_matrix(true_values, predicted_values), '\n',
          'report:', metrics.classification_report(true_values, predicted_values)
          )


print_metrics(y_test, y_lr)
print_metrics(y_test, y_bg)
print_metrics(y_test, y_rf)
print_metrics(y_test, y_xgb)
# lr < bg = rf = xgb



