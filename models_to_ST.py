# -*- coding: utf-8 -*-
"""
interpreter:py 3.5
data: financial data of some public companies
model: use classification model to predict whether a public company would be punished by ST(Special Treatment)
and indicate the hidden risk of investing a certain company
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn import metrics
from keras import Sequential
from keras.layers.core import Dense, Activation
from sklearn.tree import DecisionTreeClassifier
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
print(lr.coef_)
# coefficient:
# [[-0.08912578 -0.74096554 -0.24405551 -1.11179539  1.09896285 -0.01494734]]
# ARA,ASSET,ATO,ROA,GROWTH,LEV,SHARE,ST

# start build LM neural network model
net = Sequential()
net.add(Dense(units=10, input_dim=6))
net.add(Activation('relu'))
net.add(Dense(units=1, input_dim=10))
net.add(Activation('sigmoid'))
net.compile(loss='binary_crossentropy', optimizer='adam')
net.fit(X_train, y_train, epochs=10, batch_size=1)
predictions2 = net.predict_classes(X_test)
print("Confusion Matrix2: \n", metrics.confusion_matrix(y_test, predictions2))
# 预测结果和LR完全一样

