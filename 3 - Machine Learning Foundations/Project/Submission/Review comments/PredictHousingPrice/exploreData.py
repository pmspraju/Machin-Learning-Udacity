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
            
            npRM = np.array(self.features['RM'])
            npLSTAT = np.array(self.features['LSTAT'])
            npPTRATIO = np.array(self.features['PTRATIO'])
            npLabel = np.array(self.label)
            
            print np.unique(npRM[np.where((npRM > 7) & (npRM < 8))])
            rm1 = np.where((npRM > 8) & (npRM < 9))
            rm1Lab = npLabel[rm1]
            print np.mean(rm1Lab)
            
            rm2 = np.where((npLSTAT > 3) & (npLSTAT < 4))
            rm2Lab = npLabel[rm2]
            print np.mean(rm2Lab)
            
            rm3 = np.where((npPTRATIO > 12) & (npPTRATIO < 13))
            rm3Lab = npLabel[rm3]
            print np.mean(rm3Lab)
            
            
            print np.unique(npLSTAT)
            print np.unique(npPTRATIO)
            
        #change panda series to numpy array
            nparr = np.array(self.label)
        
        #Minimum price of the data        
            #self.minimum_price = nparr.min()
            self.minimum_price = np.min(nparr)

        #Maximum price of the data
            #self.maximum_price = nparr.max()
            self.maximum_price = np.max(nparr)

        #Mean price of the data
            #self.mean_price = nparr.mean()
            self.mean_price = np.mean(nparr)

        #Median price of the data
            self.median_price = np.median(nparr)

        #Standard deviation of prices of the data
            #self.std_price = nparr.std()
            self.std_price = np.std(nparr)
        
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
