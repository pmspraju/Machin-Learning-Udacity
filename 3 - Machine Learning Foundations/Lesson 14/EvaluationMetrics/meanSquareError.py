# -*- coding: utf-8 -*-
"""
Created on Mon May 22 07:31:54 2017

@author: Madhu
"""

import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn import cross_validation
X_train, X_test, y_train, y_test = (
        cross_validation.train_test_split(X, 
                                        y, 
                                        test_size=0.4, 
                                        random_state=0)
                                    )
reg1 = DecisionTreeRegressor()
reg1.fit(X_train, y_train)
print "Decision Tree mean absolute error: {:.2f}".format(mse(y_test, reg1.predict(X_test)))

reg2 = LinearRegression()
reg2.fit(X_train, y_train)
print "Linear regression mean absolute error: {:.2f}".format(mse(y_test, reg2.predict(X_test)))

results = {
 "Linear Regression": mse(y_test, reg2.predict(X_test)),
 "Decision Tree": mse(y_test, reg1.predict(X_test))
}