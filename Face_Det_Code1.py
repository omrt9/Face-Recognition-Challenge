import cv2
import os
from keras.preprocessing.image import load_img
import numpy as np
from keras.preprocessing import image

import numpy as np


files = os.listdir(os.getcwd() + '/Data/')
IMG_SIZE=100

data_img = []
label = []
for f in files:
    img = cv2.imread(os.getcwd() + '/Data/' + f,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = np.array(img)
    img=img.flatten()
    img=img.reshape((1,10000))
    temp_label = f.split('_')[0][:9]
    data_img.append((img))
    label.append((temp_label))
    

data_img = np.array(data_img)
data_img = data_img.reshape((584,10000))
label = np.array(label)
label = label.reshape((584,1))
np.savez('NN_data.npz',X=data_img,y=label)