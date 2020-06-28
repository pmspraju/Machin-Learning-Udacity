# -*- coding: utf-8 -*-
"""
Created on Sun May 28 05:49:56 2017

@author: Madhu
"""
# Import libraries necessary for this project
import os
import pandas as pd

class loadData(object):
    
    def __init__(self,path,filename):
        self.path = path
        self.filename = filename
        
    def load(self):
        try:
            #change the working directory
            os.chdir(self.path)
            # Load the Boston housing dataset
            self.data = pd.read_csv(self.filename)
            return self.data
        except:
            print("Exception in class:loadData.py method:load")
        
    def desc(self):
        try:
            print self.data.describe()
            
        except:
            print("Exception in class:loadData.py method:desc")
        