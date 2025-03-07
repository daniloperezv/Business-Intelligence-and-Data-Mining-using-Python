##TRAINING MODELS - PERFORMANCE MEASUREMENT - LOGISTIC REGRESSION##
###################################################################

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:27:38 2025

@author: danil
"""

##Logistic regression with stats
import numpy as np
from sklearn import datasets
import statsmodels.formula.api as sm

#We load the iris dataset
iris = datasets.load_iris()
X=iris["data"][:,3]  #petal width
y=(iris["target"]==2).astype(np.int)

#We reshape X and Y
X=X.reshape(-1,1)
y=y.reshape(-1,1)

#We run the model
model=sm.Logit(y, X)

#We obtain the results of the model using the model.fit option 
result=model.fit()

#We show the summary of the results
result.summary()

#%%

##Logistic regression with schikit-learn
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

#We load the iris dataset
iris = datasets.load_iris()
X=iris["data"][:,3]  #petal width
y=(iris["target"]==2).astype(np.int)

#We reshape X and Y
X=X.reshape(-1,1)
y=y.reshape(-1,1)

#We run the model
log_reg=LogisticRegression()

#We obtain the results of the model using the fit option 
log_reg.fit(X,y)

#We have the coefficients
coefficients=log_reg.coef_

#We have the predicted values
preds=log_reg.predict(X)

#We have the probabilities
probabilities=log_reg.predict_proba(X)

#%%

##Goodness-of-fit metrics for classification models
from sklearn import metrics

#Accuracy Score
accuracy_score=metrics.accuracy_score(y,preds)
print(accuracy_score)

#Confusion Matrix
cf=metrics.confusion_matrix(y,preds)
print(cf)

#Recall Score (Sensitivity)
recall_score=metrics.recall_score(y,preds,average=None)

#Precision Score (Specificity)
precision_score=metrics.precision_score(y,preds,average=None)

#Classification Report (Overall)
report=metrics.classification_report(y,preds)
print(report)

#ROC Curve (Receiver Operating Characteristic)
from sklearn.metrics import roc_curve, auc
fpr, tpr, thrs = roc_curve(y[:, 0], probabilities[:, 1])
#fpr: false positive rate; tpr: true positive rate

#ROC AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr) 

# Compute ROC curve and ROC area for each class using matplolib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve(area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
