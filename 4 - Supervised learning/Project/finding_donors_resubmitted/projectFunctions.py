# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:20:49 2017

@author: Madhu
"""
import os
import pandas as pd
import numpy  as np
import visuals as vs
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import make_scorer
import time
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#Function to load the data
def loadData(path,filename):
    try:
            #change the working directory
            os.chdir(path)
            # Load the Boston housing dataset
            data = pd.read_csv(filename)
            return data
    except:
            print "-----------------------------------------------------------------------"
            print("Exception in function:loadData.py")

#Function to explore the data
def exploreData(data):
    try:
           #Total number of records                                  
           rows = data.shape[0]
           cols = data.shape[1]          
           n_records = rows
           incomeArr = np.array(data['income'])

           #Number of records where individual's income is more than $50,000           
           ind = np.where(incomeArr == '>50K')[0]
           n_greater_50k = ind.size

           #Number of records where individual's income is at most $50,000           
           ind = np.where(incomeArr == '<=50K')[0]
           n_at_most_50k = ind.size

           # TODO: Percentage of individuals whose income is more than $50,000
           greater_percent = 100 * float(n_greater_50k)/float(n_records)

           # Print the results
           print "-----------------------------------------------------------------------"
           print "Total number of records: {}".format(n_records)
           print "Individuals making more than $50,000: {}".format(n_greater_50k)
           print "Individuals making at most $50,000: {}".format(n_at_most_50k)
           print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)
    except:
           print "-----------------------------------------------------------------------"
           print("Exception in function:exploreData.py")
           
#Transform the skewed features
def transformData(data):
    try:
        #Tranform the skewed featrues using logarthmic tranformation
        # Split the data into features and target label        
        features_raw = data.drop('income', axis = 1)
        income_raw = data['income']
        #display(features_raw.head(n=1))
        # Visualize skewed continuous features of original data        
        vs.distribution(data)
        
        # Log-transform the skewed features
        skewed = ['capital-gain', 'capital-loss']
        features_log_transformed = pd.DataFrame(data = features_raw)
        
        #display(features_raw.head(n=1))
        features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
        
        # Visualize the new log distributions        
        vs.distribution(features_log_transformed, transformed = True)
        
        #Transform the numerical features
        # Initialize a scaler, then apply it to the features
        scaler = MinMaxScaler() # default=(0, 1)
        numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

        features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
        features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
        
        # Show an example of a record with scaling applied
        display(features_log_minmax_transform.head(n = 5))
        
        # TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
        features_final = pd.get_dummies(features_log_minmax_transform)
        
        #display(features_final.head(n = 5))
        
        # TODO: Encode the 'income_raw' data to numerical values
        income = pd.get_dummies(income_raw)
        #print(income['>50K'])

        # Print the number of features after one-hot encoding
        encoded = list(features_final.columns)
        print "{} total features after one-hot encoding.".format(len(encoded))

        # Uncomment the following line to see the encoded feature names
        #print encoded
        return features_final, income
        
    except:
        print("Exception in function:transformData.py")

#split the data in to train and test data
def splitData(features_transformed,income_transformed):
    try:
        # Split the 'features' and 'income' data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features_transformed,
                                                    income_transformed, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

        # Show the results of the split
        print "Training set has {} samples.".format(X_train.shape[0])
        print "Testing set has {} samples.".format(X_test.shape[0])
        return X_train, X_test, y_train, y_test
    except:
        print("Exception in function:transformData.py")

#calculate metrics
def metricData(income_actual,income_predicted):
    try:
        print(len(income_actual))
        print(len(income_predicted))
    except:
        print("Exception in function:transformData.py")

def trainPredict(learner, sample_size, X_train, y_train, X_test, y_test):
    try:
        results = {}
        
        start_time = time.clock()
        clf_fit_train = learner.fit(X_train[:sample_size], y_train[:sample_size])
        end_time = time.clock()
        results['train_time'] = end_time - start_time
               
        start_time = time.clock()
        clf_predict_test = clf_fit_train.predict(X_test)
        clf_predict_train = clf_fit_train.predict(X_train[:sample_size])
        end_time = time.clock()
        results['pred_time'] = end_time - start_time  
               
        results['acc_train'] = accuracy_score(y_train[:sample_size], clf_predict_train)
        results['acc_test'] = accuracy_score(y_test, clf_predict_test)
        results['f_train'] = fbeta_score(y_train[:sample_size], clf_predict_train, average='binary', beta=0.5)
        results['f_test'] = fbeta_score(y_test, clf_predict_test, average='binary', beta=0.5)
        print results        
        return results       
    except Exception as e:
        print("Exception in function trainPredict.py"+ str(e))
        
def learnModels(X_train, y_train, X_test, y_test,accuracy, fscore):
    try:
        clf_gnb = GaussianNB()
        clf_dt = DecisionTreeClassifier(random_state=0)
        clf_svm = svm.SVC(random_state=0)
        
        #Calculate the number of samples for 1%, 10%, and 100% of the training data
        #samples_100 is the entire training set i.e. len(y_train)
        #samples_10 is 10% of samples_100
        #samples_1 is 1% of samples_100
        train_len = len(X_train)
        samples_100 = train_len
        samples_10 = int(round(train_len/10,0))
        samples_1 = int(round(train_len/100,0))
        
        # Collect results on the learners
        results = {}
        for clf in [clf_gnb, clf_dt, clf_svm]:
            clf_name = clf.__class__.__name__
            print(clf_name)
            results[clf_name] = {}
            for i, samples in enumerate([samples_1, samples_10, samples_100]):
                results[clf_name][i] = \
                trainPredict(clf, samples, X_train, y_train, X_test, y_test)                
                #print(results[clf_name][3])
        # Run metrics visualization for the three supervised learning models chosen        
        vs.evaluate(results, accuracy, fscore)        
    except Exception as e:
        print("Exception in function learnModels.py"+ str(e))

def tuneModel(X_train,y_train,X_test,y_test):
    try:
        print("coluns")
        print(X_train.columns.values)
        clf = DecisionTreeClassifier(random_state=0)
        params = {'criterion':['gini','entropy'], 'max_depth' : np.array([6,7,8])}
        scoring_fnc = make_scorer(fbeta_score,beta=0.5)
        grid = GridSearchCV(clf, params, scoring=scoring_fnc)
        grid_prm = grid.fit(X_train,y_train)
        best_clf = grid_prm.best_estimator_
        best_predictions = best_clf.predict(X_test)
        pred = (clf.fit(X_train, y_train)).predict(X_test)
        print "Unoptimized model\n------"
        print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, pred))
        print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, pred, beta = 0.5))
        print "\nOptimized Model\n------"
        print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
        print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))
    except Exception as e:
        print("Exception in function tuneModels.py"+ str(e))

def randomForestclassifier(X_train,y_train,X_test,y_test):
    try:
       clf = RandomForestClassifier(max_depth=2, random_state=0)
       rf_clf = clf.fit(X_train,y_train)
       importances = clf.feature_importances_
       predictions = rf_clf.predict(X_test)
       
       print "Random classifier accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, predictions))
       print "Random classifier F-score on the testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5))
       vs.feature_plot(importances, X_train, y_train)
    except Exception as e:
        print("Exception in function tuneModels.py"+ str(e))