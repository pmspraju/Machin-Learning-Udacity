# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:14:26 2017

@author: Madhu
"""
# Import libraries necessary for this project
import os
import sklearn
import matplotlib.pyplot as plt
# Import supplementary visualizations code visuals.py
import visuals as vs
import seaborn as sns; sns.set()
import pandas as pd
#import numpy as np
from IPython.display import display # Allows the use of display() for DataFrames
import numpy as np

print "-----------------------------------------------------------------------"
print('The scikit-learn version is {}.'.format(sklearn.__version__))
#get the working directory and filename
path = os.getcwd()
filename = "customers.csv"

#load data using load class and print describe of data
from projectFunctions import loadData,featureRelevance,testNormality,logScaling,removeOutliers,pcaImplement,gmmCluster,dataRecovery
data = loadData(path,filename)

print "-----------------------------------------------------------------------"
names = list(data)
# Success - Display the first record
display(data.head(n=1))
#print names
#print len(names) - 1
#display(data.describe())

#feature relevance. Select a feature and findout the prediction accuracy
feature = 'Grocery'
score = featureRelevance(data,feature)
print "Removed and predicted feature:" + feature
print "Final accuracy score on the testing data: {:.4f}".format(score)

#create subplots
axes = []
fig = plt.figure(figsize = (10,3))

cnt = 0
for i in names:
    cnt = cnt +1
    axes.append(fig.add_subplot(1,len(names),cnt, title=i))
    bins,det = testNormality(data[i])
    axes[cnt-1].plot(bins, det)

#plt.setp(axes, yticks=[])
plt.tight_layout(h_pad=50)
plt.show()

pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


#axes.append(fig.add_subplot(1,3,1))
f, axes = plt.subplots(1, 1, figsize=(7, 7))
cor = data.corr()
ax = sns.heatmap(cor,ax=axes)
plt.show()

#Scaling
log_data = logScaling(data)

#removing outliers
outliers_features = removeOutliers(log_data)
outliers  = [65,66,154]
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

#implement pca
pca =pcaImplement(good_data)
#print pd.DataFrame(pca.components_,columns=good_data.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6'])

#visualize the principal components
pca_results = vs.pca_results(good_data, pca)

#reduce the data
reduced_data = pca.transform(good_data)
# Create a DataFrame for the reduced data
reduced_data1 = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

#visualize the biplot
vs.biplot(good_data,reduced_data1,pca)

#gaussian mixture model algorithm
[preds,centers,sh_score] = gmmCluster(reduced_data)
print "silhouette score for GMM: {:.4f}".format(sh_score)
#pca_samples = np.array([[-0.77325655, -2.43331189],[-1.6607017,   1.26456724],[ 3.31630556, -0.1736468 ]])
#pca_samples = pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2'])
#vs.cluster_results(reduced_data1, preds, centers, pca_samples)

#recover original center points
true_centers = dataRecovery(data,pca,centers)
display(true_centers)