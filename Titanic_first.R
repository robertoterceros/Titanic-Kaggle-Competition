#importing useful libraries
library("class")
library("mice")
library("rpart")
library("rpart.plot")

#--------------------- Helper Function ----------------------------------#
csv_submision <- function(filename, predicted_value, labels){
  output = list()
  output$PassengerId = labels
  output$Survived = predicted_value
  write.table(output, file = filename,row.names=FALSE, sep = ",")
}

# calculating accuracy from confusion matrix
accuracy <- function(confusion_matrix){
  accuracy = (confusion_matrix[1,1] + confusion_matrix[2,2])/(confusion_matrix[1,1] + confusion_matrix[2,2] + confusion_matrix[1,2] + confusion_matrix[2,1])
  return(accuracy)
}

#Importing the data
setwd("~/Kaggle Competition/Titanic")

titanic_train = read.csv("train.csv", header = TRUE)
titanic_test = read.csv("test.csv", header = TRUE)

labels = titanic_test$PassengerId #Save the labels in order to create the submission file 

#----------------------------------------- Cleaning the data -----------------------------------------#
#SEX
#Replace female by 0 and male by 1
titanic_test$Sex = factor(ifelse(titanic_test$Sex=="female",0,1)) 
titanic_train$Sex = factor(ifelse(titanic_train$Sex=="female", 0,1))

#EMBARKED
#dealing with missing values from Embarked (2 values missing in ID = 62, 830)
titanic_train$Embarked[c(62,830)] <- "S"
#plot(titanic_train$Embarked,titanic_train$Survived)
#Since I detected some relationship between the C and survival (and Q,S and not survival) I will transform in quantitative value
sum(titanic_train$Survived[titanic_train$Embarked=="C"])/sum(titanic_train$Embarked=="C")
sum(titanic_train$Survived[titanic_train$Embarked=="Q"])/sum(titanic_train$Embarked=="Q")
sum(titanic_train$Survived[titanic_train$Embarked=="S"])/sum(titanic_train$Embarked=="S")
titanic_train$Embarked = as.numeric(ifelse(titanic_train$Embarked=="C",0,ifelse(titanic_train$Embarked=="Q",1,2)))
titanic_test$Embarked = as.numeric(ifelse(titanic_test$Embarked=="C",0,ifelse(titanic_test$Embarked=="Q",1,2)))

#AGE
#from the histogram I decided to model the Age with t student distribution with 
mu_Age = mean(titanic_train$Age, na.rm=TRUE)
sigma_Age = sd(titanic_train$Age, na.rm=TRUE)

for(i in 0:length(titanic_train$Age)){
  if(is.na(titanic_train$Age[i]) == TRUE) {
    titanic_train$Age[i] = min( abs( rt(1, 1) * sigma_Age + mu_Age), 70)
  }}

####add variable child?
titanic_train$Child <- 0
titanic_train$Child[titanic_train$Age < 18] <- 1
TitanicTree = rpart(Survived ~Pclass + Sex + SibSp + Parch + Child, data = titanic_train, method="class")
prp(TitanicTree)

#--------------------------- LOGISTIC REGRESSION MODEL -----------------------------------------#

#building the model
logitMod = glm(Survived ~ Sex + Pclass + SibSp, data = titanic_train, family = binomial(link="logit"))
summary(logitMod)

# Making predictions on training set
predictTrain1 = predict(logitMod, type="response")
summary(predictTrain1)
table1 <- table(predictTrain1>0.5, titanic_train$Survived)
#####0.8002245
accuracy(table1)

#prediction on the test set
predictTest = predict(logitMod, type="response", newdata=titanic_test)
output = as.numeric(ifelse(predictTest>0.5, 1, 0))
#save file in order to submit
csv_submision("submission.csv", output, labels)

#########################################SECOND FORM #################################
#missing values in Age
library(mice)
# No iteration. But I want to get Predictor-Matrix
init = mice(titanic_train, maxit=0) 
predM = init$predictorMatrix
# Do not use following columns to impute values in 'Age'. Use the rest.
predM[, c("PassengerId", "Name","Ticket","Cabin")]=0    
imp<-mice(titanic_train, m=5, predictorMatrix = predM)
# Get the final data-frame with imputed values filled in 'Age'
tr <- complete(imp)
View(tr)

logitMod2 = glm(Survived ~ Sex + Pclass  + SibSp + Age, data = tr, family = binomial(link="logit"))
summary(logitMod2)
predictTrain2 = predict(logitMod2, type = "response")
summary(predictTrain2)
table(predictTrain2>0.5,tr$Survived)

#------------------------------------ TREE MODEL (cart IN R) ---------------------------------------------------#
# CART model
Titanic_Tree = rpart(Survived ~ Pclass + Sex + SibSp + Parch + Embarked, data = titanic_train, method="class", minbucket=30)
prp(Titanic_Tree)
predictTitanic = predict(Titanic_Tree, newdata = titanic_train, type = "class")
confusion_treee = table(predictTitanic, titanic_train$Survived)
print(accuracy(confusion_treee))
#0.8114478

# We will use k-fold cross-validation in order to decide what value of minbucket we take:
myvars = c("Survived", "Pclass", "Sex", "SibSp", "Parch", "Embarked")
myvars2 = c("Pclass", "Sex", "SibSp", "Parch", "Embarked")

titanic_train = titanic_train[myvars]
titanic_test = titanic_test[myvars2]

folds = createFolds(titanic_train$Survived, k = 10)

cv = lapply(folds, function(x) {
  training_fold = titanic_train[-x,]
  test_fold = titanic_train[x,]
  titanic_tree <- rpart(Survived ~ Pclass + Sex + SibSp + Parch + Embarked, 
                        data = training_fold, 
                        method="class", 
                        minbucket=30)
  predictTitanic = predict(titanic_tree, newdata = test_fold, type = "class")
  cm = table(predictTitanic, test_fold['Survived'][,1])
  print(cm)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))
print(accuracy)
#0.8114732



titanic_tree <- rpart(Survived ~ Pclass + Sex + SibSp + Parch + Embarked, 
                      data = titanic_train, 
                      method="class", 
                      minbucket=30)
predictTitanic = predict(titanic_tree, newdata = titanic_test, type = "class")

csv_submision("submission8.csv",predictTitanic, labels)
  

#on test set
predictTitanic = predict(TitanicTree, newdata = titanic_test, type = "class")
output3 = list()
output3$PassengerId = titanic_test$PassengerId
output3$Survived = as.numeric(predictTitanic)-1
write.table(output3, file = "submission3.csv",row.names=FALSE, sep = ",")

#Let's replace na in Age by a number between 21 and 29
titanic_train$Age[is.na(titanic_train$Age)] <- sample(21:29, 1)
TitanicTree = rpart(Survived ~Pclass + Sex + SibSp + Parch + Age, data = titanic_train, method="class", minbucket=25)
prp(TitanicTree)
predictTitanic = predict(TitanicTree, newdata = titanic_train, type = "class")
table(predictTitanic, titanic_train$Survived)
#0 494  98
#1  55 244
# (494+244)/(98+494+55+244) =0.8282828
predictTitanic = predict(TitanicTree, newdata = titanic_test, type = "class")
output3b = list()
output3b$PassengerId = titanic_test$PassengerId
output3b$Survived = as.numeric(predictTitanic)-1
write.table(output3b, file = "submission3b.csv",row.names=FALSE, sep = ",")




#Lets create a training and test set (500,391) split of the 891 observations, grow the tree on the training set, and evaluate its performance on the test set.

#-------------------------------------- SVM --------------------------------------------#

library(e1071)
classifier = svm(Survived ~Pclass + Sex + SibSp + Parch + Age, 
                 data = titanic_train,
                 type = "C-classification",
                 kernel = "polynomial")
y_pred = predict(classifier, newdata = titanic_train)
table(y_pred, titanic_train$Survived)
#titanic test age:
titanic_test$Age[is.na(titanic_test$Age)] <- sample(21:29, 1)


predictTitanic = predict(classifier, newdata = titanic_test, type = "class")
output4 = list()
output4$PassengerId = titanic_test$PassengerId
output4$Survived = as.numeric(predictTitanic)-1
write.table(output4, file = "submission4.csv",row.names=FALSE, sep = ",")


#Predicting the test


#------------------------ k nearest neighborn with K-fold cross ---------------------------#
myvars = c("Survived", "Pclass", "Sex", "SibSp", "Parch", "Embarked")
myvars2 = c("Pclass", "Sex", "SibSp", "Parch", "Embarked")

titanic_train = titanic_train[myvars]
titanic_test = titanic_test[myvars2]

folds = createFolds(titanic_train$Survived, k = 10)

cv = lapply(folds, function(x) {
  training_fold = titanic_train[-x,]
  test_fold = titanic_train[x,]
  knn_temp <- knn(training_fold[myvars2], test_fold[myvars2], training_fold$Survived, k = 17)
  cm = table(knn_temp, test_fold['Survived'][,1])
  print(cm)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))
print(accuracy)

#In order to select the best value for the k-fold
accu <- numeric(30-4)
for(iteration in 5:30) {
  cv = lapply(folds, function(x) {
    training_fold = titanic_train[-x,]
    test_fold = titanic_train[x,]
    knn_temp <- knn(training_fold[myvars2], test_fold[myvars2], training_fold$Survived, k = iteration)
    cm = table(knn_temp, test_fold['Survived'][,1])
    print(cm)
    accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
    return(accuracy)
  })
  accuracy = mean(as.numeric(cv))
  accu[iteration] = accuracy
}
lines(accu[5:30])

plot(accu[5:30])

# with k=1 Nearest neighborn we obtained 0.7789014
# with k=5                              0.78
# with k=10                             0.7878652
# with k =15                            0.79
#The best seems to be 17 (or arount)

knn.17 <- knn(titanic_train[myvars2], titanic_test, titanic_train$Survived, k = 17)
csv_submision("submission7.csv", knn.17, labels)

#-------------------------------I Will use the hole dataset to predict the trainig data

labels = titanic_test["Survived"]
labels = labels[,1]

tr = titanic_train[myvars]
cv = titanic_test[myvars]

tr$Sex = factor(ifelse(tr$Sex=="female",0,1)) #Replace female by 0 and male by 1
cv$Sex = factor(ifelse(cv$Sex=="female", 0,1))

knn.final <- knn(tr, cv, labels, k = 20)

output4 = list()
output4$PassengerId = titanic_test$PassengerId
output4$Survived = as.integer(knn.final)-1

write.table(output4, file = "submission4.csv",row.names=FALSE, sep = ",")


# IDEAS:
#--------------------10 fold k-fold cross validation-------------------#
# Random forest
#change split the data into train and cross-validation


