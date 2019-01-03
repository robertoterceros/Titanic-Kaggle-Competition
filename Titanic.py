# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 18:08:19 2018

@author: Roberto
"""
%reset
# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

# Feature Engineering
y = dataset["Survived"]
X = dataset.drop("Survived", axis = 1)
X['Has_Cabin'] = X["Cabin"].apply(lambda x: 0 if type(x) == float else 1) #Determine if customer had a cabin
X = dataset.drop(["Cabin", "Ticket"], axis = 1) #Get rid of the whole attribute


X["Age"].fillna(X["Age"].mean(), inplace = True)
X["Embarked"].fillna(method = 'backfill', inplace = True)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
X["Sex"] = encoder.fit_transform(X["Sex"])
X["Embarked"] = encoder.fit_transform(X["Embarked"])
#onehotencoder = OneHotEncoder()
#X = onehotencoder.fit_transform(X["Embarked"]).toarray()

#Feature Engineering with the X["Name"] we will try to obtain titles
X_Name = X["Name"]
X_title = list()

def get_title(X_Name, X_title):
    for name in X_Name:  
        if '.' in name:
            X_title.append(name.split(',')[1].split('.')[0].strip())
        else:
            X_title.append('Unknown')
    return X_title

get_title(X_Name, X_title)
X["title"] = X_title
X["title"] = encoder.fit_transform(X["title"])
X = X.drop(["Name"], axis = 1)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv = 20)
accuracies.mean()
accuracies.std()

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# NOW FOR THE TEST SET
test_X = test_dataset.drop(["Cabin", "Ticket"], axis = 1) #Get rid of the whole attribute

test_X["Age"].fillna(test_X["Age"].mean(), inplace = True)
test_X["Embarked"].fillna(method = 'backfill', inplace = True)
test_X["Fare"].fillna(method = 'backfill', inplace = True)

encoder = LabelEncoder()
test_X["Sex"] = encoder.fit_transform(test_X["Sex"])
test_X["Embarked"] = encoder.fit_transform(test_X["Embarked"])
#onehotencoder = OneHotEncoder()
#X = onehotencoder.fit_transform(X["Embarked"]).toarray()

#Feature Engineering with the X["Name"] we will try to obtain titles
test_X_Name = test_X["Name"]
test_X_title = list()
get_title(test_X_Name, test_X_title)
test_X["title"] = test_X_title

test_X["title"] = encoder.fit_transform(test_X["title"])
test_X = test_X.drop(["Name"], axis = 1)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 12, criterion = 'entropy', random_state = 0)
classifier.fit(X, y)
y_pred_test = classifier.predict(test_X)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)











# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,4:5])
X[:, 4:5] = imputer.transform(X[:,4:5])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,9] = labelencoder_X.fit_transform(X[:,9])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()