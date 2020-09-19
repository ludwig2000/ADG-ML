import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from logistic_regression_tests import LogisticRegression

bc=pd.read_csv('/home/ritwik/Desktop/Workspace/ADG-ML/WEEK-2/heart.csv')
x=bc["chol"]
x1=np.array(x)
X=x1.reshape(-1,1)
y=bc.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)

def accuracy(y_true,y_pred):
    accuracy=np.sum(y_true==y_pred)/len(y_true)
    return accuracy

reg=LogisticRegression(lr=0.0001,n_iters=5000)
reg.fit(X_train,y_train)
predictions=reg.predict(X_test)

print("LR accuracy: ",accuracy(y_test,predictions))