# -*- coding: utf-8 -*-
"""
Created on Mon May 29 05:45:33 2017

@author: Madhu
"""
from sklearn import cross_validation

class dataSplit(object):

    def __init__(self,features,label,test_size):
        self.features = features
        self.label = label
        self.test_size = test_size
        
    def trainTestSplit(self):
        
        try:
            #call train_test_split
            [self.features_train, self.features_test, self.label_train, self.label_test] = (
                    cross_validation.train_test_split(self.features, 
                                                      self.label, 
                                                      test_size=self.test_size, 
                                                      random_state=2)
                                                                         )
            print "Training and testing split was successful."
            return [self.features_train, self.features_test, self.label_train, self.label_test]         
        
        except:
            print "Error in class:dataSplit method:trainTestSplit"
            return [0,0,0,0]
        
    def printLen(self):
        
        try:
            print "Out of {} features, {} split as training set".format(len(self.features),len(self.features_train))
            print "Out of {} features, {} split as test set".format(len(self.features),len(self.features_test))
            print "Out of {} labels, {} split as training set".format(len(self.label),len(self.label_train))
            print "Out of {} features, {} split as test set".format(len(self.label),len(self.label_test))
        
        except:
            print "Error in class:dataSplit method:printLen"