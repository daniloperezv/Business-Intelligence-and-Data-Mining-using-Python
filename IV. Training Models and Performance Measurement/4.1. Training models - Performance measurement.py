##TRAINING MODELS - PERFORMANCE MEASUREMENT - LINEAR REGRESSION##
#################################################################

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:40:11 2025

@author: danilo
"""

##Linear regression with numpy
from __future__ import division
import numpy as np

#We create a vector of random X 
X=np.random.rand(100,1)

#We define my theta parameters to be found 
theta0=2.5
theta1=5

#Given X and the parameters, we can calculate the Yreal
Yreal=theta0+theta1*X

#Typically, we observe a noise or measurement error in real life
noise=np.random.randn(100,1)
y=Yreal+noise

#Our vector of X will have a constant
constant=np.ones((100,1))

#We concatenate the X with the constant
X_b=np.concatenate((constant,X),axis=1)

#Now, we run the regression and after we conduct a performance measurement using different metrics
theta_best=np.dot(np.dot(np.linalg.inv(np.dot(X_b.T,X_b)),X_b.T),y)
y_est=np.dot(X_b,theta_best)

y_prom=y_est.mean()

m=len(X_b)
n=X_b.shape[1]

#R2
r2=(np.dot(np.dot(theta_best.T,X_b.T),y)-m*y_prom**2)/(np.dot(y.T,y)-m*y_prom**2)
r2theil=1-(1-r2)*(m-1)/(m-n)
r2goldberg=(1-(n/m))*r2

#F-test
f_test=(r2/(n-1))/((1-r2)/(m-n))

#We calculate the variance and the standard error
v=np.var(y-y_est)
v_theta=np.linalg.inv(np.dot(X_b.T,X_b))*v
errorsd_theta=np.sqrt(np.diag(v_theta)).reshape(len(v_theta),1)

#ttest
t_test=theta_best/errorsd_theta
desvest=np.sqrt(v)
#%%

##Linear Regression with stats
import statsmodels.api as smf

#We create a vector of random X 
X=np.random.rand(100,1)

#We define my theta parameters to be found 
theta0=2.5
theta1=5

#Given X and the parameters, we can calculate the Yreal
Yreal=theta0+theta1*X

#Typically, we observe a noise or measurement error in real life
noise=np.random.randn(100,1)
y=Yreal+noise

#Our vector of X will have a constant
constant=np.ones((100,1))

#We run the regression
est=smf.OLS(y,X_b).fit()

#And we print the summary of the estimation
print(est.summary())

#%%

##Linear Regression with scikit-learn

#First, we import the command LinearRegression from the library sklearn.linear_model 
from sklearn.linear_model import LinearRegression

#We run the regression directly
reg=LinearRegression(fit_intercept=False)
reg=reg.fit(X_b,y)

#We obtained the thetas
thetas=reg.coef_

#And the predicted values (preds)
preds=reg.predict(X_b)

#Now, we obtain the metrics using the scikit-learn library
mae=metrics.mean_absolute_error(y,preds)
mse=metrics.mean_squared_error(y,preds)
rmse=mse**0.5
r2=metrics.r2_score(y,preds)



