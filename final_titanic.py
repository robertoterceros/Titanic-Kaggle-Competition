# -*- coding: utf-8 -*-
"""
FINAL TITANIC
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
y = dataset["Survived"]
X = dataset.drop(["Survived"], axis = 1)

X_new = pd.concat([X,test_dataset])

X_new = X_new.drop(["Ticket"], axis = 1) #Get rid of the whole attribute
#X_new['Has_Cabin'] = X_new["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
X_new["Age"].fillna(X_new["Age"].mean(), inplace = True)
X_new["Embarked"].fillna(method = 'backfill', inplace = True)
X_new["Fare"].fillna(method = 'backfill', inplace = True)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
X_new["Sex"] = encoder.fit_transform(X_new["Sex"])
X_new["Embarked"] = encoder.fit_transform(X_new["Embarked"])
#onehotencoder = OneHotEncoder()
#X = onehotencoder.fit_transform(X["Embarked"]).toarray()

#Feature Engineering with the X["Name"] we will try to obtain titles
X_Name = X_new["Name"]
X_title = list()

def get_title(X_Name, X_title):
    for name in X_Name:  
        if '.' in name:
            X_title.append(name.split(',')[1].split('.')[0].strip())
        else:
            X_title.append('Unknown')
    return X_title

get_title(X_Name, X_title)
X_new["title"] = X_title

rare_title = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']

X_new['title'].replace('Mlle', 'Miss', inplace = True)
X_new['title'].replace('Ms', 'Miss', inplace = True)
X_new['title'].replace('Mme', 'Mrs', inplace = True)


X_new["title"] = encoder.fit_transform(X_new["title"])
X_new = X_new.drop(["Name","Cabin"], axis = 1)

X_train = X_new[X_new["PassengerId"] < 892]
X_test = X_new[X_new["PassengerId"] > 891]

# RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y)
y_pred_test = classifier.predict(X_test)

my_dict = {'Survived': y_pred_test}
my_dict['PassengerId'] = X_test['PassengerId']

solution = pd.DataFrame(my_dict, columns = ['PassengerId','Survived'])
solution.to_csv('submission.csv', index = False)


true_solution = pd.read_csv('solution.csv')
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_solution['Survived'], y_pred_test)



