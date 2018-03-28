# -*- coding: utf-8 -*-
"""
interpreter:py 3.5
data: financial data of some public companies
model: use logistic regression to predict whether a public company would be punished by ST(Special Treatment)
and indicate the hidden risk of investing a certain company
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn import metrics

# read the data
finan_data = pd.read_csv('./input/财务报表分析与ST预测.csv', encoding='utf-8', index_col=0)
print(finan_data.head())
# data description
print(finan_data.describe())
X_train, X_test, y_train, y_test = train_test_split(finan_data.ix[:, :-1], finan_data.ix[:, -1],
                                                    test_size=0.3, random_state=0)
print(y_train.head())

# start build LR model
lr = LR()
lr.fit(X_train, y_train)
print('训练集上的正确率：', lr.score(X_train, y_train))

# test the model
predictions = lr.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print("AUC: ", metrics.roc_auc_score(y_test, predictions))
print("Confusion Matrix: \n", metrics.confusion_matrix(y_test, predictions))
print(metrics.classification_report(y_test, predictions))
# 模型的准确率达到了95%， 可以说是非常好了，这与数据的选取有关，用的数据是已经做好特征工程的数据，所以效果很好
