# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 09:23:42 2018

@author: hp
"""

import numpy as np
import cv2
from keras.models import load_model
from sklearn import decomposition
from numpy.linalg import pinv
import re
import glob


seed = 7
np.random.seed(seed)

shuffle_data = True  # shuffle the addresses before saving
hdf5_path = 'Face_dataset.hdf5'  
data_path = 'E:/AAASem-6/ML/Face_Recog/Data/ML face images/*.jpg'
addrs = glob.glob(data_path)

TRAIN_DIR = 'E:\AAASem-6\ML\Face_Recog\Data\ML face images'
IMG_SIZE = 10000
LR = 1e-3
l= []
for i in addrs:
    l.append(re.findall('\d+',i)[1]);

labels=[]
cnt=0;
labels.append(cnt);
for i in range(1,len(l)):
    if(l[i]==l[i-1]):
        labels.append(cnt);
    else:
        cnt=cnt+1;
        labels.append(cnt);
        
dictonary=dict(zip(labels,l))
model=load_model('train_100000.h5')


    
test='E:/AAASem-6/ML/Face_Recog/Data/ML face images\\201501031_glasses_O.jpeg'
img=cv2.imread(test,cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(1,IMG_SIZE))
#img=img/255;
eig_vec = np.load('eig_vector.npz')
eig_vec = eig_vec['eig']

eig_vec_inv = pinv(eig_vec)

print(img.shape)
print(eig_vec_inv.shape)

img1 = np.dot(eig_vec, img)

print(img1.shape)

b=model.predict_classes(img1.T)
print(b)
print(dictonary[int(b)])
