# -*- coding: utf-8 -*-
"""
Created on Sun May 28 07:46:32 2017

@author: Madhu
"""
# Import libraries necessary for this project
import numpy as np

class exploreData(object):

    def __init__(self,features,label):
        self.features = features
        self.label = label
        
    def calcStat(self):
        
        try:
            
        #change panda series to numpy array
            nparr = np.array(self.label)
        
        #Minimum price of the data        
            self.minimum_price = nparr.min()

        #Maximum price of the data
            self.maximum_price = nparr.max()

        #Mean price of the data
            self.mean_price = nparr.mean()

        #Median price of the data
            self.median_price = np.median(nparr)

        #Standard deviation of prices of the data
            self.std_price = nparr.std()
        
        #return statistics to main
            return [self.minimum_price,self.maximum_price,self.mean_price,self.median_price,self.std_price]
        
        except:
            print("Exception in class:exploreData.py method:calcStat")
            return [0,0,0,0,0]
    
    def printStat(self):
        try:    
            # Show the calculated statistics
            print "Statistics for Boston housing dataset:\n"
            print "Minimum price: ${:,.2f}".format(self.minimum_price)
            print "Maximum price: ${:,.2f}".format(self.maximum_price)
            print "Mean price: ${:,.2f}".format(self.mean_price)
            print "Median price ${:,.2f}".format(self.median_price)
            print "Standard deviation of prices: ${:,.2f}".format(self.std_price)
            
        except:
            print("Exception in class:exploreData.py method:printStat") 
