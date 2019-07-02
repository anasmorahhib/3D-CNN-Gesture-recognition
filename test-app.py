# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 20:40:23 2019
@author: santonas
"""


#%%
import tensorflow as tf
print(tf.__version__)
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from time import sleep

#%%
classes = [
    "Swiping Right",
    "Sliding Two Fingers Left",
    "No gesture",
    "Thumb Up"
    ]
#%%
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

#%%
def normaliz_data(np_data):
    # Normalisation
    scaler = StandardScaler()
    #scaled_images  = normaliz_data2(np_data)
    scaled_images  = np_data.reshape(-1, 30, 64, 64, 1)
    return scaled_images

#%%
def normaliz_data2(v):
    normalized_v = v / np.sqrt(np.sum(v**2))
    return normalized_v


#%%

class Conv3DModel(tf.keras.Model):
  def __init__(self):
    super(Conv3DModel, self).__init__()
    # Convolutions
    self.conv1 = tf.compat.v2.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last')
    self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
    self.conv2 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last')
    self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2), data_format='channels_last')
    self.convLSTM =tf.keras.layers.ConvLSTM2D(40, (3, 3))
    #self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
    #self.conv3 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', name="conv3", data_format='channels_last')
    #self.pool3 = tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3), data_format='channels_last')
    #norm
    self.flatten =  tf.keras.layers.Flatten(name="flatten")

    # Dense layers
    self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
    self.out = tf.keras.layers.Dense(4, activation='softmax', name="output")
    

  def call(self, x):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.convLSTM(x)
    #x = self.pool2(x)
    #x = self.conv3(x)
    #x = self.pool3(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.out(x)

#%%
new_model = Conv3DModel()
#%%
new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop())

#%%
new_model.load_weights('weights/path_to_my_weights2')

#%%
to_predict = []
num_frames = 0
cap = cv2.VideoCapture(0)
classe =''

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    to_predict.append(cv2.resize(gray, (64, 64)))
    
         
    if len(to_predict) == 30:
        frame_to_predict = np.array(to_predict, dtype=np.float32)
        frame_to_predict = normaliz_data(frame_to_predict)
        #print(frame_to_predict)
        predict = new_model.predict(frame_to_predict)
        classe = classes[np.argmax(predict)]
        
        print('Classe = ',classe, 'Precision = ', np.amax(predict)*100,'%')


        #print(frame_to_predict)
        to_predict = []
        #sleep(0.1) # Time in seconds
        #font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, classe, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),1,cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()