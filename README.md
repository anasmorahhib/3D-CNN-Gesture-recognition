# 3D-CNN-Gesture-recognition
Gesture recognition via 3D CNN neural network using Tensorflow.  
Our project consists in developing a Python language solution, using deep learning techniques, for hand gestures recognition.
the idea of this project is to detect the gestures in a three-dimensional space,
 that is to say that, instead of analyzing the shape of the hand in each image separately, we will analyze the movement of the hand in time,
 which means analyzing several images at once.   
 A little example to understand what we want to do,
If you raise two fingers, the machine must know it, but the 3D gives us more than that, it will be able to detect if we move the two fingers left or right or other,
 which makes us analysis the gesture and the movement of the hand too.
 
 ### The technologies used:
> Python 3.6  
TensorFlow 2.0 Beta  
OpenCV 3
### Description to read before you begin:
- **20BN-JESTER**: It contains all training, validation and testing data. You have to download it from [The 20BN-jester Dataset V1](https://20bn.com/datasets/jester/v1#download)

- **training-sample.py**: If you use your personal computer, often you will not have the performance needed to work with all this data.
 this script will help you bring out a sample that you can work with.
 if you want to work with all data, ignore this file.

- **validation-sample.py**: the same thing as **training-sample.py** for validation.

- **main-beginners-syntax.py**: the main code with a simple syntax. 

- **main-experts-syntax.py**: the main code with a more complex syntax.

- **test-app.py**: a small application to test the model after training.

### Noticeable:
as you know, a deep learning project goes through several steps before starting the training of the model, like the data cleaning, the standardization, the creation of the model ...
for this in both files **main-beginners-syntax.py** and **main-experts-syntax.py** it is better to use a block-by-block execution using spider for example, or use the file **main.ipynb** with jupyter notebook.  
the advantage of this is that if you got an error or you want to change in a step, you do not need to run all the code each time, just execute the part you want.  
I invite you to install Anaconda, to have a good environment.


### Demo 
Youtube link - https://youtu.be/dMAAyceGKAg

<img src="https://j.gifs.com/oVA6XX.gif" alt="Gesture recognition demo" width="500"/>