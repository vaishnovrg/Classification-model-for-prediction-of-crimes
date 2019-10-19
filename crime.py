import pandas as pd
import numpy as np

#Reading dataset
train=pd.read_csv('criminal_train.csv')
X_train=train.drop('Criminal', axis=1)
Y_train=train.iloc[:,-1].values

#Oversampling method to increase minority class of '1'
from imblearn.over_sampling import SMOTE
smt = SMOTE()
X_train, Y_train = smt.fit_sample(X_train, Y_train)

#Splitting training data
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 0)


#Boosting model to train the data
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(xTrain, yTrain)
y_pred = model.predict(xTest)


#K fold cross validation to check overfitting
from sklearn.model_selection import KFold
scores = []
best_xgb = XGBClassifier()
cv = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in cv.split(X_train):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    x_train, x_test, y_train, y_test = X_train[train_index], X_train[test_index], Y_train[train_index], Y_train[test_index]
    best_xgb.fit(x_train, y_train)
    scores.append(best_xgb.score(x_test, y_test))
    
print(np.mean(scores))    

#Confusion matrix/Accuracy
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
results = confusion_matrix(yTest, y_pred) 
print ('Confusion Matrix :')
print(results) 
print ('Accuracy Score :',accuracy_score(yTest, y_pred) )
print ('Report : ')
print (classification_report(yTest, y_pred) )

#Prediction output on test dataset
test=pd.read_csv('criminal_test.csv')
xTest=test.iloc[:,:].values
Predicted=model.predict(xTest)
