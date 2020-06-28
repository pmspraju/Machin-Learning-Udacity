# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:39:59 2017

@author: Madhu
"""
from sklearn.model_selection import ShuffleSplit
from sklearn import tree
import numpy as np
from sklearn.metrics import r2_score,make_scorer
from sklearn.grid_search import GridSearchCV
from perfMetric import perfMetric

class fitModel(object):
    
    def __init__(self,features,label):
        self.features = features
        self.label = label

    def fit_model(self):
        try:
            
            X = self.features
            y = self.label                                               
            # Create cross-validation sets from the training data
            # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
            # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
            ss = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
            cv_sets = ss.get_n_splits(X.shape[0])
             
            # TODO: Create a decision tree regressor object
            regressor = tree.DecisionTreeRegressor()
             
            # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
            params = {'max_depth' : np.array([1,2,3,4,5,6,7,8,9,10])}
             
            # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
            scoring_fnc = make_scorer(r2_score)
             
            # TODO: Create the grid search object
            grid = GridSearchCV(regressor, params, cv=cv_sets, scoring=scoring_fnc)
            
            # Fit the grid search object to the data to compute the optimal model
            grid = grid.fit(X, y)
        
            # Return the optimal model after fitting the data
            return grid.best_estimator_
        
        except:
            print "Error in class:fitModel method:fit_Model"    