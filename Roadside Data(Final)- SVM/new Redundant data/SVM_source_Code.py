# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:10:44 2020

@author: ZAdak
"""

import pandas as pd
dataset = pd.read_csv('ReduceData_Roadside.csv')
dataset.shape

import warnings
warnings.filterwarnings("ignore")

X = dataset.drop(['AQI'],axis=1)
X.head()
y = dataset['AQI']
y.head()

from sklearn.model_selection import train_test_split
X_train , X_test,y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=20)

print(' X is ', X_train.shape)
print(' X is ', X_test.shape)
print(' Y is ', y_train.shape)
print(' Y is ', y_test.shape)


from sklearn.svm import SVC
sv = SVC()
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# le.fit(X_train['RO3'].astype(str))
# X_train['RO3']= le.transform(X_train['RO3'].astype(str))
# X_test['RO3'] = le.transform(X_test['RO3'].astype(str))

# le = LabelEncoder()
# le.fit(X_train['RNXO'].astype(str))
# X_train['RNXO']= le.transform(X_train['RNXO'].astype(str))
# X_test['RNXO'] = le.transform(X_test['RNXO'].astype(str))

# le = LabelEncoder()
# le.fit(X_train['RPM25'].astype(str))
# X_train['RPM25']= le.transform(X_train['RPM25'].astype(str))
# X_test['RNXO'] = le.transform(X_test['RPM25'].astype(str))

# le = LabelEncoder()
# le.fit(X_train['RSO2'].astype(str))
# X_train['RS02']= le.transform(X_train['RS02'].astype(str))
# X_test['RS02'] = le.transform(X_test['RSO2'].astype(str))

sv.fit(X_train,y_train)


y_pred = sv.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))