# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 08:52:01 2018

@author: hp
"""

import numpy as np
from numpy import linalg as LA
from sklearn import decomposition


train_data=np.load('train_data_10000.npz');
X=train_data['X']
y=train_data['y']
print(X.shape)
m=X.shape
X=X.reshape((m[0],m[3]))
print(X.shape)

#PCA

pca = decomposition.PCA(0.95,whiten=True)
pca.fit(X)
X = pca.transform(X)
print(X.shape)
cov = pca.get_covariance()
print(cov.shape)
print(pca.components_)
np.savez('eig_vector.npz',eig=pca.components_)

print(pca.explained_variance_ratio_)


np.savez('train_data_PCA_10000.npz', X=X,y=y)