# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 08:59:49 2018

@author: hp
"""

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from keras.models import load_model
from sklearn import decomposition
import cv2

train_data=np.load('NN_data.npz');
X=train_data['X']/255
X=X.reshape((584,10000))
y=train_data['y']
y=y.reshape((584,1))
print(X.shape)
print(y.shape)

#PCA

pca = decomposition.PCA(0.97,whiten=True)
pca.fit(X)
X = pca.transform(X)



encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

def base_model():
    model=Sequential()
    model.add(Dense(232, input_dim=232, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(41, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model=base_model()
model.fit(X, dummy_y, epochs=50, batch_size=16)
model.save('train_100000.h5') 
scores = model.evaluate(X, dummy_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


IMG_SIZE=100

    
test='E:/AAASem-6/ML/Data/201501109_anger.jpg'
img=cv2.imread(test,cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
img = np.array(img)
img=img.flatten()
img=img.reshape((1,10000))
img=img/255;
eig_vec = np.load('eig_vector.npz')
eig_vec = eig_vec['eig']

img1 = pca.transform(img)

print(img1.shape)

b=model.predict_classes(img1)
print(b)