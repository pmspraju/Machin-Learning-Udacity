# -*- coding: utf-8 -*-
"""
Created on Mon May 29 05:01:27 2017

@author: Madhu
"""
from sklearn.metrics import r2_score

class perfMetric(object):

    def __init__(self,y_true,y_predict):
        self.y_true = y_true
        self.y_predict = y_predict
        
    def r2Score(self):
        
        try:
            #call r2_score
            self.score = r2_score(self.y_true, self.y_predict)
            return self.score
        
        except:
            print "Error in class:perMetric method:r2Score"
            return 0
    
    def printScore(self):
            print "r2_score: ${:,.3f}".format(self.score) 
            