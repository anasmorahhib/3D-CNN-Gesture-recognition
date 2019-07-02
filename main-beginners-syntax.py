#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


# In[3]:


from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())


# In[4]:


import os
import math
import pandas as pd
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[5]:


tf.__version__


# In[6]:


# return gray image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


# In[7]:


# training targets, you can use your custome csv file if you already created it with "training-sample.py"
targets = pd.Series.from_csv('data_csv/jester-v1-train.csv',header=None,sep = ";").to_dict()
targets[34870]
# validation targets, you can use your custome csv file if you already created it with "validation-sample.py"
targets_validation = pd.Series.from_csv('data_csv/jester-v1-validation.csv',header=None,sep = ";").to_dict()
targets_validation[9223]


# In[8]:


len(targets)
targets[100000]


# In[9]:


# classes label you want to use all labels 
'''label = pd.read_csv('data_csv/labels.csv',header=None, usecols=[0])
label.head()
targets_name = label[0].tolist()
len(targets_name)'''

# The classes (labels) we want to use
targets_name = [
    "Swiping Right",
    "Sliding Two Fingers Left",
    "No gesture",
    "Thumb Up"
    ]


# In[11]:


# Get the data directories
path = "training_samples/"
path_cv = "validation_samples/"

dirs = os.listdir(path)
dirs_cv = os.listdir(path_cv)


# In[12]:


# number of samples for training and validation
print(len(dirs))
print(len(dirs_cv))


# In[13]:


training_count = 10


# In[14]:


'''
The videos do not have the same number of frames, here we try to unify.
'''
hm_frames = 30 # number of frames
# unify number of frames for each training
def get_unify_frames(path):
    offset = 0
    # pick frames
    frames = os.listdir(path)
    frames_count = len(frames)
    # unify number of frames 
    if hm_frames > frames_count:
        # duplicate last frame if video is shorter than necessary
        frames += [frames[-1]] * (hm_frames - frames_count)
    elif hm_frames < frames_count:
        # If there are more frames, then sample starting offset
        #diff = (frames_count - hm_frames)
        #offset = diff-1 
        frames = frames[0:hm_frames]
    return frames  


# In[15]:


# Resize frames
def resize_frame(frame):
    frame = img.imread(frame)
    frame = cv2.resize(frame, (64, 64))
    return frame
    


# In[16]:


# Adjust training data
counter_training = 0 # number of training
training_targets = [] # training targets 
new_frames = [] # training data after resize & unify
for directory in dirs:
    new_frame = [] # one training
    # Frames in each folder
    frames = get_unify_frames(path+directory)
    if len(frames) == hm_frames: # just to be sure
        for frame in frames:
            frame = resize_frame(path+directory+'/'+frame)
            new_frame.append(rgb2gray(frame))
            if len(new_frame) == 15: # partition each training on two trainings.
                new_frames.append(new_frame) # append each partition to training data
                training_targets.append(targets_name.index(targets[int(directory)]))
                counter_training +=1
                new_frame = []


# In[17]:


# we do the same for the validation data
counter_validation = 0
cv_targets = []
new_frames_cv = []
for directory in dirs_cv:
    new_frame = []
    # Frames in each folder
    frames = get_unify_frames(path_cv+directory)
    if len(frames)==hm_frames:
        for frame in frames:
            frame = resize_frame(path_cv+directory+'/'+frame)
            new_frame.append(rgb2gray(frame))
            if len(new_frame) == 15:
                new_frames_cv.append(new_frame)
                cv_targets.append(targets_name.index(targets_validation[int(directory)]))
                counter_validation +=1
                new_frame = []


# In[18]:


# To check training length
print(len(new_frames))
print(len(training_targets))


# In[19]:


# To check validation length
print(len(new_frames_cv))
print(len(cv_targets))


# In[20]:


training_targets[0:20]


# In[21]:


#show data
fig = plt.figure()
for i in range(2,4):
    for num,frame in enumerate(new_frames[i][0:18]):
        y = fig.add_subplot(4,5,num+1)
        y.imshow(frame, cmap='gray')
    fig = plt.figure()
plt.show()


# In[22]:


# convert training data to np float32
training_data = np.array(new_frames[0:counter_training], dtype=np.float32)


# In[23]:


# Function to empty the RAM
def release_list(a):
   del a[:]
   del a


# In[24]:


release_list(new_frames)


# In[26]:


# convert validation data to np float32
cv_data = np.array(new_frames_cv[0:counter_validation], dtype=np.float32)


# In[27]:


release_list(new_frames_cv)


# In[25]:


training_data.shape


# In[28]:


cv_data.shape


# In[29]:


# Normalisation: training
print('old mean', training_data.mean())
scaler = StandardScaler()
scaled_images  = scaler.fit_transform(training_data.reshape(-1, 15*64*64))
print('new mean', scaled_images.mean())
scaled_images  = scaled_images.reshape(-1, 15, 64, 64, 1)
print(scaled_images.shape)


# In[30]:


# Normalisation: validation
print('old mean', cv_data.mean())
scaler = StandardScaler()
scaled_images_cv  = scaler.fit_transform(cv_data.reshape(-1, 15*64*64))
print('new mean',scaled_images_cv.mean())
scaled_images_cv  = scaled_images_cv.reshape(-1, 15, 64, 64, 1)
print(scaled_images_cv.shape)


# In[32]:


# My model
class Conv3DModel(tf.keras.Model):
  def __init__(self):
    super(Conv3DModel, self).__init__()
    # Convolutions
    self.conv1 = tf.compat.v2.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last')
    self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
    self.conv2 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last')
    self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2), data_format='channels_last')
   
    # LSTM & Flatten
    self.convLSTM =tf.keras.layers.ConvLSTM2D(40, (3, 3))
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


# ## 1. the simplest method with beginner syntax

# In[33]:


model = Conv3DModel()


# In[ ]:


# choose the loss and optimizer methods
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])


# In[35]:


x_train = np.array(scaled_images)
y_train = np.array(training_targets)
x_val = np.array(scaled_images_cv)
y_val = np.array(cv_targets)


# In[36]:


print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)


# #### here I use the checkpoints
# read more:
# https://www.tensorflow.org/beta/guide/checkpoints

# In[37]:


# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_today/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True)


# In[ ]:


# Run the training 
history = model.fit(x_train, y_train,
                    callbacks = [cp_callback],
                    validation_data=(x_val, y_val),
                    batch_size=32,
                    epochs=10)


# In[39]:


# just after one epoch
history.history


# In[41]:


# save the model for use in the application
model.save_weights('weights/path_to_my_weights', save_format='tf')