# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


linear=pd.read_csv("input/linear-regression-dataset/Linear Regression - Sheet1.csv")
#remove the print when you finish this is just to visualize data 


linear=linear.iloc[:297]
#First we create  our figure to plot in 
fig = plt.figure()

ax = fig.add_subplot(1,1,1)


ax.plot(linear['X'], linear['Y'])
ax.set_xlabel('input - x')
ax.set_ylabel('target - y')
plt.show()


X = linear.iloc[:,0:1].values
Y = linear.iloc[:,-1].values
Y=np.array(Y).reshape(-1,1)

X_train, X_test,y_train,y_test=train_test_split(X,Y,test_size=0.05,random_state=2)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)
y_test=pd.DataFrame(y_test)


(m,n)=X_train.shape
theta=np.zeros((n,1))

def   cost(X_train,y_train,theta) :
    J=0
    h= X_train@theta
    sqrErrors = (h - y_train)**2
    J = 1/(2*m) * sqrErrors.sum()
    return  J 
J=cost(X_train,y_train,theta)


iterations=100

alpha=0.00002

def DescentGradient(X_train,y_train,theta,num_iters,alpha):
     
    cost_history = []
#   stocker les valeurs de theta dans une liste
    past_thetas = [theta]
    for i in range(num_iters):
        h= X_train@theta
        delta = (X_train.transpose()@(h - y_train))/m
        theta= theta- alpha*delta
        cost_i= cost(X_train,y_train,theta)
        cost_history.append(cost_i)
        past_thetas.append(theta)
        

    return past_thetas,cost_history
theta_hist,J_hist=DescentGradient(X_train,y_train,theta,iterations,alpha)
theta=theta_hist[-1]

x = np.arange(1,len(J_hist)+1) #iterations
plt.plot(x,J_hist, '*', linewidth=1, markersize=8)
plt.xlabel("num_iters")
plt.ylabel("Cost Function")
plt.show()

lr_pred =X_test@theta
#print(lr_pred)

r2_lr = r2_score(y_test, lr_pred)*100
mae_lr = mean_absolute_error(y_test, lr_pred)
mse_lr = mean_squared_error(y_test, lr_pred)
print([r2_lr, mae_lr, mse_lr])
plt.plot(lr_pred)
plt.plot(y_test)
fig, ax = plt.subplots()
ax.plot([0,1],[0,1], transform=ax.transAxes)

plt.scatter(lr_pred, y_test, s=500, alpha=1,marker=r'$\clubsuit$',label="Luck",facecolor='black')
plt.xlabel("Prediction")
plt.ylabel("Observation")

plt.show()