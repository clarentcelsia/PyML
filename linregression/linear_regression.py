# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 22:59:22 2022

@author: Allen
"""

#REGRESSION 
    # This model searches for relationships among variables.
    # Linear regression often used to predict the value of variable based on the value of another variable, i.e
    # In medical research uses linear regression to understand the relationship between drug dosage and blood pressure of patients.

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def linear_regression(X, y):
    linreg = LinearRegression()
    linreg.fit(X,y) # >>> optimal value of b0, b1 [y = b0 + b1.x]
    
    # Check the obtained coef b0, b1
    b0 = linreg.intercept_ # >>> model predicts the response 8.11 when 𝑥 is zero
    b1 = linreg.coef_ # >>> the predicted response rises by -0.094 when 𝑥 is increased by one.
    
    result = linreg.score(X,y) # >>> return the coefficient of determination of the prediction
                               # a measure that assesses the ability of a model to predict or explain an outcome in the linear regression setting.
    
    # Predict
    y_pred = linreg.predict(X)
    print(y_pred)
    
    # Visualization
    plt.figure()
    plt.plot(X, y_pred, color='r')
    plt.scatter(X, y, marker=".")
    

def poly_regression(X,y):
    # Fit the input array (x) and (x^degree) into one statement
    x_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X,y) # >>> i.e [[5.0, 25.0],[8.0, 64.0],..]
    
    linear = LinearRegression().fit(x_poly, y)
    
    # Check the ability of model to predict with poly data
    pred_accuracy = linear.score(x_poly, y) # >>> 0.59 / 59%
    
    b0 = linear.intercept_
    b1 = linear.coef_ 
    
    # Predict
    y_pred = linear.predict(x_poly)
    print(y_pred)
    
    plt.figure()
    plt.plot(X, y_pred, color='r')
    plt.scatter(X, y, marker=".")

def lin_poly_stat(X,y):
    # You need to add the column of ones to the inputs (x) if you want statsmodels to calculate the intercept 𝑏₀. 
    # It doesn’t take 𝑏₀ into account by default.
    x_new = sm.add_constant(X)
    
    # https://www.statsmodels.org/stable/regression.html
    # Create model then fit in
    model = sm.OLS(y,X)
    result = model.fit()
    print(result.summary())
    
    # Check the ability of model to predict
    pred_coef = result.rsquared # >>> 0.95

    y_pred = result.fittedvalues
    print(y_pred)
    
def open_file(file):
    data = pd.read_csv(file)
    X = data.iloc[:,1].values
    y = data.iloc[:,2].values
    return X,y
    
if __name__=="__main__":
    #x = np.array([-2,3,5,12,15,21]).reshape(-1,1) # >>> 1 column, 6 rows
    #y = np.array([2,4,4,5,7,12]).reshape(-1,1)
    
    X, y = open_file('C:\Clarenti\Data\Project\ML\Python\Basic\Code\linregression\Doc\Data.csv')
    X_ = X.reshape(-1,1)
    y_ = y.reshape(-1,1)
    
    linear_regression(X_, y_)
    poly_regression(X_, y_)
    lin_poly_stat(X_, y_)
    
    