from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import cv2
import os
import math

batch_size = 32
nb_classes = 8
nb_epoch = 20
numimages=1000
# input image dimensions
img_rows, img_cols = 512, 512
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (4, 4)
# convolution kernel size
kernel_size = (5, 5)
def listFiles(path, extension):  
    return [path+f for f in os.listdir(path) if f.endswith(extension)]  

image1 = listFiles('/home/chaksouv1008/Desktop/kaggle/train/ALB/','.jpg')
image2 = listFiles('/home/chaksouv1008/Desktop/kaggle/train/BET/','.jpg')
image3 = listFiles('/home/chaksouv1008/Desktop/kaggle/train/DOL/','.jpg')
image4 = listFiles('/home/chaksouv1008/Desktop/kaggle/train/LAG/','.jpg')
image5 = listFiles('/home/chaksouv1008/Desktop/kaggle/train/NoF/','.jpg')
image6 = listFiles('/home/chaksouv1008/Desktop/kaggle/train/OTHER/','.jpg')
image7 = listFiles('/home/chaksouv1008/Desktop/kaggle/train/SHARK/','.jpg')
image8 = listFiles('/home/chaksouv1008/Desktop/kaggle/train/YFT/','.jpg')
def load_data(image):
  result=np.zeros((1,img_rows,img_cols))
  #b=math.floor(len(image)/10)
  #image=image[:int(b)]
  for i in range(0,len(image)):
    p=str(i) + "/" + str(len(image))
    print(p)
    i=image[i]
    x=cv2.resize(cv2.imread(i,cv2.IMREAD_GRAYSCALE),(img_cols,img_rows))
    result=np.append(result,np.array([x]),axis=0)
  result=result[1:,:,:]
  result=result[np.random.choice(result.shape[0], numimages,replace='T'),:,:]
  return result
xxx=np.append(load_data(image1),load_data(image2),axis=0)
xxx=np.append(xxx,load_data(image3),axis=0)
xxx=np.append(xxx,load_data(image4),axis=0)
xxx=np.append(xxx,load_data(image5),axis=0)
xxx=np.append(xxx,load_data(image6),axis=0)
xxx=np.append(xxx,load_data(image7),axis=0)
xxx=np.append(xxx,load_data(image8),axis=0)
yyy=np.append(np.ones(numimages),2*np.ones(numimages))
yyy=np.append(yyy,3*np.ones(numimages))
yyy=np.append(yyy,4*np.ones(numimages))
yyy=np.append(yyy,5*np.ones(numimages))
yyy=np.append(yyy,6*np.ones(numimages))
yyy=np.append(yyy,7*np.ones(numimages))
yyy=np.append(yyy,8*np.ones(numimages))
yyy=yyy-1
print(xxx.shape)
print(yyy.shape)
# the data, shuffled and split between train and test sets
def dataa(xxx,yyy):
  indx=np.random.choice(len(yyy), len(yyy),replace='T')
  xt=xxx[indx]
  yt=yyy[indx]
  iii=int(math.floor(len(xt)*0.7))
  xtr=xt[0:iii,:,:]
  ytr=yt[0:iii]
  xts=xt[iii:,:,:]
  yts=yt[iii:]
  return ((xtr,ytr),(xts,yts))


(X_train, y_train), (X_test, y_test) = dataa(xxx,yyy)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255.0
X_train = X_train/255.0
X_test /= 255.0
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
#model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=pool_size))


model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("yup!")
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
