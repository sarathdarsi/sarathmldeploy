# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 12:01:48 2021

@author: Bharath
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC 
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import time
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


# Loading the data 

data =  pd.read_csv('C:\AIMLTraining\sensor.csv')


# Data cleaning

data.info()

data.isnull().sum()

# deleting column named as sensor 15.
data = data.drop('sensor_15', 1)
data = data.drop('Unnamed: 0', 1)
data.shape

data['machine_status'].value_counts()

#creating new columns named as data and time from timestamp and deleting the column timestamp.
data['date'] = data['timestamp'].apply(lambda x: x.split(' ')[0])
data['time'] = data['timestamp'].apply(lambda x: x.split(' ')[1])
data = data.drop(['timestamp'], 1)

# imputting missing values with median of each column
data_imputed = data.fillna(data.median())


#checking out histogram for outlier data
data_imputed.hist(figsize=(15,15))

#removing outliers using zscore
z_scores = zscore(data_imputed.iloc[:,:51])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data_outlir = data_imputed[filtered_entries]

# how much data has been lost after removing outliers
((220320 - 165201)/220320) * 100

data_outlir['machine_status'].value_counts()


# standarizing the dataset using minmax scalar
scaler = MinMaxScaler()
data_std = scaler.fit_transform(data_outlir.iloc[:,:51])
data_std = pd.DataFrame(data_std)


# checking for variance of each feature
data_va = data_std.var(axis= 0)
data_vas = data_va.sort_values(ascending=False)
y = data_vas.values 
x = range(len(y))
plt.figure(figsize = (20,20))
plt.plot(x, y)
plt.show()

corrmatrix = data_std.corr()

# doing multicollinearity test
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    return dataset


le = LabelEncoder()
Y = le.fit_transform(data_outlir['machine_status'])

X = correlation(data_std, 0.7)
X

# Explodatory data analysis
for i in data_std.columns:
    plt.scatter( data_outlir['machine_status'] , data_std[i] )
    plt.xlabel('machine_status')
    plt.ylabel(i)
    plt.show()


plt.scatter( data_outlir['time'] , data_outlir['machine_status'] )
plt.xlabel('time')
plt.ylabel('machine status')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

knn = KNeighborsClassifier(n_neighbors=15)
clf = knn.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_knb_model=roc_auc_score(y_test, y_pred)*100
acc_knb_model


#applying logistic regression
lr = LogisticRegression(C = 0.2)
clf1 = lr.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)
acc_log_reg=roc_auc_score(y_test, y_pred1)*100
acc_log_reg



#applying naive bayes
clf2 = GaussianNB().fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)
acc_nb=roc_auc_score(y_test, y_pred2)*100
acc_nb


#applying decision tree
clf3 = tree.DecisionTreeClassifier().fit(X_train, y_train)
y_pred3 = clf3.predict(X_test)
acc_dt=roc_auc_score(y_test, y_pred3)*100
acc_dt



#applying random forest
clf4 = RandomForestClassifier(max_depth=5, random_state=0).fit(X_train, y_train)
y_pred4 = clf4.predict(X_test)
acc_rmf_model=roc_auc_score(y_test, y_pred4)*100
acc_rmf_model



#applying support vector machine
clf5 = SVC(gamma='auto').fit(X_train, y_train)
y_pred5 = clf5.predict(X_test)
acc_svm_model=roc_auc_score(y_test, y_pred5)*100
acc_svm_model

#applying Stochastic Gradient descent

sgd_model=SGDClassifier()
sgd_model.fit(X_train,y_train)
sgd_pred=sgd_model.predict(X_test)
acc_sgd=round(sgd_model.score(X_train,y_train)*100,10)
acc_sgd



