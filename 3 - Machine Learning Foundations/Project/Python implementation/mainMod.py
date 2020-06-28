# -*- coding: utf-8 -*-
"""
Created on Sun May 28 07:05:03 2017

@author: Madhu
"""
# Import libraries necessary for this project
import os
import sklearn
# Pretty display for notebooks
#%matplotlib inline

from loadData import loadData
print "-----------------------------------------------------------------------"
print('The scikit-learn version is {}.'.format(sklearn.__version__))
#get the working directory and filename
path = os.getcwd()
filename = "housing.csv"

#load data using load class and print describe of data
lobj = loadData(path,filename)
data = lobj.load()
print "-----------------------------------------------------------------------"
lobj.desc()
print "-----------------------------------------------------------------------"

prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)
print "-----------------------------------------------------------------------"

#explore the data
from exploreData import exploreData

stat = exploreData(features,prices)
[minimum_price,maximum_price,mean_price,median_price,std_price] = stat.calcStat()
stat.printStat()
print "-----------------------------------------------------------------------"

#Developing a model

#import performance metric class
from perfMetric import perfMetric

#calculate score with test values
pm = perfMetric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
score = pm.r2Score()
pm.printScore()
print "-----------------------------------------------------------------------"

#shuffle and split
from dataSplit import dataSplit

#create object of the imported class
test_size = 0.2
split = dataSplit(features,prices,test_size)
X_train, X_test, y_train, y_test = split.trainTestSplit()
split.printLen()
print "-----------------------------------------------------------------------"
#Learning curves
# Import supplementary visualizations code visuals.py
import visuals as vs

#decision tree learning curves for different max_depths
vs.ModelLearning(features, prices)

#Complexity curves for decision tree model
vs.ModelComplexity(X_train, y_train)

print "-----------------------------------------------------------------------"
#fitting a model
from fitModel import fitModel

#fm = fitModel(features,prices)
fm = fitModel(X_train,y_train)
best_estimator = fm.fit_model()

print "Parameter 'max_depth' is {} for the optimal model.".format(best_estimator.get_params()['max_depth'])
print "-----------------------------------------------------------------------"

# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(best_estimator.predict(client_data)):
    print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)
    
print "-----------------------------------------------------------------------"

    
#check score for train data set
y_true = y_train
y_predict = best_estimator.predict(X_train)
#calculate score with test values
pm = perfMetric(y_true, y_predict)
score = pm.r2Score()
print "Train dataset score: ${:,.3f}".format(score)

#check score for test data set
y_true = y_test
y_predict = best_estimator.predict(X_test)
#calculate score with test values
pm = perfMetric(y_true, y_predict)
score = pm.r2Score()
print "Test dataset score: ${:,.3f}".format(score)

#check score for full data set
y_true = prices
y_predict = best_estimator.predict(features)
#calculate score with test values
pm = perfMetric(y_true, y_predict)
score = pm.r2Score()
print "Full dataset score: ${:,.3f}".format(score)
print "-----------------------------------------------------------------------"





