# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 23:30:49 2017

@author: Madhu
"""

# ----------
# 
# In this exercise, you will add in code that decides whether a perceptron will fire based
# on the threshold. Your code will go in lines 32 and 34. 
#
# ----------
import numpy as np

class Perceptron:
    """
    This class models an artificial neuron with step activation function.
    """
    def __init__(self, weights = np.array([1]), threshold = 0):
        """
        Initialize weights and threshold based on input arguments. Note that no
        type-checking is being performed here for simplicity.
        """
        self.weights = weights
        self.threshold = threshold
    
    def activate(self,inputs):
        """
        Takes in @param inputs, a list of numbers equal to length of weights.
        @return the output of a threshold perceptron with given inputs based on
        perceptron weights and threshold.
        """ 
        # The strength with which the perceptron fires.
        strength = np.dot(self.weights, inputs)
        # TODO: return 0 or 1 based on the threshold
        if strength <= self.threshold :
            self.result = 0
        else:
            self.result = 1   
        return self.result

    def update(self, values, train, eta=.1):
        """
        Takes in a 2D array @param values consisting of a LIST of inputs and a
        1D array @param train, consisting of a corresponding list of expected
        outputs. Updates internal weights according to the perceptron training
        rule using these values and an optional learning rate, @param eta.
        """
        # For each data point:
        for data_point in xrange(len(values)):
            # TODO: Obtain the neuron's prediction for the data_point --> values[data_point]
            prediction = self.activate(values[data_point])
            # Get the prediction accuracy calculated as (expected value - predicted value)
            # expected value = train[data_point], predicted value = prediction
            error = train[data_point] - prediction
            # TODO: update self.weights based on the multiplication of:
            # - prediction accuracy(error)
            # - learning rate(eta)
            # - input value(values[data_point])
            weight_update = eta * error * values[data_point]
            self.weights = np.float_(self.weights)
            self.weights += weight_update    
    
    def printTh(self,n):
        if n == 0:
            print "thrshold is zero:disregarded"
        else:
            print "threshold is > zero:considered"