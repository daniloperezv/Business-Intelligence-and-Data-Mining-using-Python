##LINEAR REGRESSION##
#####################

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 01:24:12 2025

@author: danilo
"""

##Linear Regression using numpy

#First, we import the library numpy and we call it np
import numpy as np

#We create a vector X of random variables
X=np.random.rand(100,1)

#We define our theta parameters to be searched  
theta0=-2.5
theta1=2

#Now, we can calculate the Yreal given X and the theta parameters
Yreal=theta0+theta1*X

#And typically, we observe a noise or measure error in real world
noise=np.random.randn(100,1)/1
Yobs=Yreal+noise


#My vector X will have a constant, so first we define it
constant=np.ones((100,1))

#Now, we concatenate the vector X with the constant
Xs=np.concatenate((constant,X),axis=1)

#We obtain the estimated thetas
theta_est=np.dot(np.dot(np.linalg.inv(np.dot(Xs.T,Xs)),Xs.T),Yobs)

#And the predicted Y (Ypred)
Ypred=np.dot(Xs,theta_est)

#%%

##Linear Regression using scikit-learn

#First, we import the command LinearRegression from the library sklearn.linear_model 
from sklearn.linear_model import LinearRegression

#We run the regression directly
reg=LinearRegression(fit_intercept=False)
reg=reg.fit(Xs,Yobs)

#We obtained the thetas
thetas=reg.coef_

#And the predicted Y (Ypred_skl)
Ypred_skl=reg.predict(Xs)



