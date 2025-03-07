##LOGISTIC REGRESSION AND GRADIENT DESCENT##
############################################
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:19:15 2025

@author: danilO
"""

#First, we import the library numpy, the datasets from scikit-learn, and the command StandardScaler
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

#We load the datasets of iris
iris=datasets.load_iris()

#We define a list
list(iris.keys())
print(iris.DESCR)

#Now, we define vectors X and Y
X=iris["data"]
Y=iris["target"]

#The vectyor X will have a constant
constant=np.ones((len(X),1))

#We concatenate the X with the constant 
X=np.concatenate((constant,X),axis=1)

#We normalize the X to make the Gradient Descent to converge much faster  
scaler=StandardScaler()
X=scaler.fit_transform(X)

#We transform the vector Y to a binary variable: 1 if it is virginic, and 0 otherwise 
Y=(Y==2).astype(np.int)

#Next, we create a logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# m = Number of rows in the matrix X
# n = Number of columns in the matrix X
m, n = X.shape

#We initialize the thetas
theta = np.zeros(n)

#We define my costs function J, which is equivalent to the SEC
def costJ(h, y):
    return (-(y * np.log(h) + (1 - y) * np.log(1 - h))).mean()

eta=0.01 #We define the learning rate
n_iteractions=10000 #We define the number of interactions "epochs" the model will do
costs=[] #Here we will be saving the error or costs function

#Now, we will be optimizing the weights using the gradient descent
for i in range(n_iteractions):
    z = np.dot(X, theta)
    h = sigmoid(z)
    
    #We calculate the error
    error = h - Y
    
    #We calculate the gradient
    gradient = X.T.dot(error)
    
    #We update the parameters
    theta=theta-eta*gradient
    cost=costJ(h,Y)
    print(i,cost)
    costs.append(cost)

#We define the predicted values
def pred(h,threshold):
    res=np.where(h >= threshold, 1, 0)
    return res
preds=pred(h,0.5)

#We import the command metrics from the scikit-learn library
from sklearn import metrics

#Now, we obtain the Accuracy Score
acc=metrics.accuracy_score(Y,preds)
print("My accuracy level was: ", acc)

#We import the library pandas
import pandas as pd

#We define the costs as a dataframe
costs=pd.DataFrame(costs)

#And we plot them
costs.plot(kind='line')
#%% 

##With scikit learn

#We import the LogisticRegression command from the scikit learn library
from sklearn.linear_model import LogisticRegression      

#We deine the logistic regression model
model=LogisticRegression()
model.fit(X,Y)

#We have the coefficients
coefs=model.coef_

#We import the command metrics from the scikit-learn library
from sklearn import metrics

#Now, we obtain the Accuracy Score
y_pred=model.predict(X)
acc1=metrics.accuracy_score(Y,preds)

print("with sklearn, ,my accuracy level was: ", acc1)


