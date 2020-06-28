# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:14:26 2017

@author: Madhu
"""
# Import libraries necessary for this project
import os
import sklearn
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from IPython.display import display # Allows the use of display() for DataFrames

print "-----------------------------------------------------------------------"
print('The scikit-learn version is {}.'.format(sklearn.__version__))
#get the working directory and filename
path = os.getcwd()
filename = "census.csv"

#load data using load class and print describe of data
from projectFunctions import loadData
data = loadData(path,filename)

print "-----------------------------------------------------------------------"
# Success - Display the first record
display(data.head(n=1))

#explore the data
from projectFunctions import exploreData
exploreData(data)

#Transform the skewed features, categorical data, numeric features
from projectFunctions import transformData
features_transformed,income_transformed = transformData(data)

#shuffle and split the data to create train and test datasets
from projectFunctions import splitData
X_train, X_test, y_train, y_test = splitData(features_transformed,income_transformed)

#calculate metrics for naive prediction
#from projectFunctions import metricData
income_actual = income_transformed['>50K']
income_predicted = pd.Series([1 for x in range(len(income_actual))])
#metricData(income_actual,income_predicted)
true_pos = float(len(income_actual[income_actual == 1]))
false_pos = float(len(income_actual) - true_pos)
true_neg = 0
false_neg = 0
prec = true_pos / (true_pos + false_pos)   
rec = true_pos / (true_pos + false_neg) 
acc = (true_pos + true_neg) / float(len(income_predicted))
beta = 0.5
fscore = (1 + beta**2) * ((prec*rec)/(((beta**2)*prec) + rec))
#print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(acc, fscore)
#print('metrics')
#print(true_pos)
#print(false_pos)
#print(prec)
#print(rec)
#print(acc)
#print(fscore)

#call predict functinon to predict and calculate accuracy
clf_gnb = GaussianNB()
sample_size = 100
from projectFunctions import trainPredict
sample_size = 300
results = trainPredict(clf_gnb, sample_size, X_train, y_train['>50K'], X_test, y_test['>50K'])

#from projectFunctions import learnModels
#learnModels(X_train, y_train['>50K'], X_test, y_test['>50K'],acc,fscore)

from projectFunctions import tuneModel
tuneModel(X_train, y_train['>50K'], X_test, y_test['>50K'])

from projectFunctions import randomForestclassifier
randomForestclassifier(X_train, y_train['>50K'], X_test, y_test['>50K'])