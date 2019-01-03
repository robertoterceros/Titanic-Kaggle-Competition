# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:25:38 2018

@author: Roberto
"""
# train, dev, test
%reset
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')
y = dataset["Survived"]
X = dataset
X['Has_Cabin'] = X["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
X_Age = X["Age"].mean()
X["Age"].fillna(X_Age, inplace = True)


#PCLASS ONE HOT ENCODING

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Frequency encoding for Embarked
# X['Embarked'] = encoder.fit_transform(X['Embarked']) TRY FREQUENCY ENCODING AS OF 5:55 OF "CATEGORICAL AND ORDINAL FEATURES"
encoding = dataset.groupby('Embarked').size()
encoding = encoding / len(dataset)
X['enc'] = X.Embarked.map(encoding)
X['enc'].fillna(0.722783, inplace = True) # I replaced nan with most frequent value


encoder = LabelEncoder()
X["Sex"] = encoder.fit_transform(X["Sex"])


X = X.drop(["Cabin", "Ticket", "Name", "Embarked", "PassengerId", "Survived"], axis = 1) #Get rid of the whole attribute

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))

#Adding the second hidden layer
#classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 4, epochs = 60)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

X_passengerId = test_dataset["PassengerId"]
X_datatest = test_dataset


X_datatest["Age"].fillna(X_Age, inplace = True)
X_datatest["Fare"].fillna(method = 'backfill', inplace = True)
X_datatest['Has_Cabin'] = X_datatest["Cabin"].apply(lambda x: 0 if type(x) == float else 1)


X_datatest["Sex"] = encoder.fit_transform(X_datatest["Sex"])
X_datatest['enc'] = X_datatest.Embarked.map(encoding)
X_datatest['enc'].fillna(0.722783, inplace = True) # I replaced nan with most frequent value


X_datatest = X_datatest.drop(["Cabin", "Ticket", "Name", "PassengerId", "Embarked"], axis = 1) #Get rid of the whole attribute


X_datatest = sc.transform(X_datatest)

y_pred_test = classifier.predict(X_datatest)
y_pred_test = 1*(y_pred_test > 0.5)
y_pred_test.reshape(1,418)



solution = pd.DataFrame(y_pred_test, index = X_passengerId)

solution.to_csv('submission.csv', index = True)


solution = pd.DataFrame(my_dict, columns = ['PassengerId','Survived'])
solution.to_csv('submission.csv', index = False)


true_solution = pd.read_csv('solution.csv')
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_solution['Survived'], y_pred_test)


