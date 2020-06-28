# -*- coding: utf-8 -*-
"""
Created on Sun May 13 12:26:15 2018

@author: C830587
"""
import os
import pandas as pd
import numpy  as np
#import visuals as vs
#from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#from IPython.display import display # Allows the use of display() for DataFrames
#import visuals as vs
from sklearn import mixture
from sklearn import metrics

#Function to load the data
def loadData(path,filename):
    try:
            #change the working directory
            os.chdir(path)
            # Load the Boston housing dataset
            data = pd.read_csv(filename)
            data.drop(['Region', 'Channel'], axis = 1, inplace = True)
            print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
            return data
    except:
            print "-----------------------------------------------------------------------"
            print("Dataset could not be loaded. Is the dataset missing?")
            print("Exception in function:loadData.py")

#Function for feature relevance
def featureRelevance(data,feature):
    try:
        
        features = data.drop(feature, axis = 1)
        label = data[feature]
        
        X_train, X_test, y_train, y_test = train_test_split(features,
                                                    label, 
                                                    test_size = 0.5 , 
                                                    random_state = 4)
        dtr = DecisionTreeRegressor(max_depth=5)
        model = dtr.fit(X_train,y_train)
        y_predict = model.predict(X_test)
        
        score = r2_score(y_test, y_predict)
        #score = fbeta_score(y_test, y_predict, average='binary', beta=0.5)
        #score = 'test'
        return score
        
    except Exception as e:
            print "-----------------------------------------------------------------------"
            print("Exception in function:featureRelevance.py"+ str(e))

#test normality of the data
def testNormality(array):
    try:
        
        length = len(array)
        n, bins, patches = plt.hist(array, length, normed=1)
        mu = np.mean(array)
        sigma = np.std(array)
        print('normality =', stats.normaltest(array))
        return bins, mlab.normpdf(bins, mu, sigma)
    
    except Exception as e:
            print "-----------------------------------------------------------------------"
            print("Exception in function:testNormality.py"+ str(e))

def logScaling(data):
    try:
        log_data = np.log(data)
        return log_data
    except Exception as e:
        print "-----------------------------------------------------------------------"
        print("Exception in function:logScaling.py"+ str(e))
        
def removeOutliers(data):
    try:
        for feature in data.keys():
    
            # TODO: Calculate Q1 (25th percentile of the data) for the given feature
            Q1 = np.percentile(data[feature],25)
    
            # TODO: Calculate Q3 (75th percentile of the data) for the given feature
            Q3 = np.percentile(data[feature],75)
    
            # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
            step = (Q3 - Q1) * 1.5
            good_data = data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))]
            # Display the outliers
            #print("Data points considered outliers for the feature '{}':".format(feature))
            #display(data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))])
        #print good_data.keys()
        return good_data
    except Exception as e:
        print "-----------------------------------------------------------------------"
        print("Exception in function:logScaling.py"+ str(e))
        
def pcaImplement(data):
    try:
        pca = PCA(n_components=2)
        return pca.fit(data)
    
    except Exception as e:
        print "-----------------------------------------------------------------------"
        print("Exception in function:pcaImplement.py"+ str(e))
    
        
def gmmCluster(data)  :
    try:
        lowest_bic = np.infty
        bic = []
        n_components_range = range(1, 6)
        cv_types = ['spherical', 'tied', 'diag', 'full']
        for cv_type in cv_types:
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cv_type)
                gmm.fit(data)
                bic.append(gmm.bic(data))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm
        #print best_gmm 
        clf = best_gmm
        preds = clf.predict(data)
        
        centers = np.empty(shape=(best_gmm.n_components, data.shape[1]))
        for i in range(best_gmm.n_components):
            density = stats.multivariate_normal(cov=best_gmm.covariances_[i], mean=best_gmm.means_[i]).logpdf(data)
            centers[i, :] = data[np.argmax(density)]
        
        print "number of components: {:.4f}".format(best_gmm.n_components)
        print "number of centers: {:.4f}".format(len(centers))
        #print centers
        sh_score = metrics.silhouette_score(data, preds, metric='euclidean')
        return preds,centers,sh_score
    except Exception as e:
        print "-----------------------------------------------------------------------"
        print("Exception in function:gmmCluster.py"+ str(e))

def dataRecovery(data,pca,centers):
    try:
        # TODO: Inverse transform the centers
        log_centers = pca.inverse_transform(centers)
        #print log_centers
        
        # TODO: Exponentiate the centers
        true_centers = np.exp(log_centers)
        #print true_centers
        
        # Display the true centers
        segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
        true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
        true_centers.index = segments
        return true_centers
        
    except Exception as e:
        print "-----------------------------------------------------------------------"
        print("Exception in function:dataRecovery.py "+ str(e))
