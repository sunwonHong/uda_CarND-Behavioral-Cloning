# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[video1]: /root/Desktop/result.mp4 "Video"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_real.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md contaning the explanation of my model and result

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model_real.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

After normalization, 6 convolutional layers and 6 fully connected layers were used, and the detailed network structure is shown below.

#### 2. Attempts to reduce overfitting in the model

1. Trying to prevent overfitting on the training set by using less epoch. 

2. Trying to prevent overfitting by increasing the number of parameters by stacking the network deeply.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I used images from the left and right cameras as well as the center camera, and for the left and right images I added or subtracted the angle by 0.2. Then, the image from the central camera was flipped left and right to increase the data once more.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I changed the network based on lenet and set the convolution kernel value to 3 to get a good result by making a deeper layer, reducing the image size through pooling in the middle, and activating the values through relu.

#### 2. Final Model Architecture

First, I divided the pixel value by 255 and subtracted 0.5 for normalization and zero-centered.

After that, in order to crop the non-road part of the image, the image was cropped by 50 from the top and 20 from the bottom by using the cropping function.

and then my model consisted of the following layers:

| Layer         		|     Description	        		| 
|:---------------------:|:---------------------------------:| 
| Input         		| 160x320x3 RGB image   			| 
| Convolution 3x3     	| 2x2 stride, output channel = 16 	|
| RELU					|									|
| Convolution 3x3     	| 2x2 stride, output channel = 32 	|
| RELU					|									|
| Convolution 3x3     	| 2x2 stride, output channel = 48 	|
| RELU					|									|
| Convolution 3x3     	| 1x1 stride, output channel = 64 	|
| RELU					|									|
| Convolution 3x3     	| 1x1 stride, output channel = 128 	|
| RELU					|									|
| Convolution 3x3     	| 1x1 stride, output channel = 128 	|
| RELU					|									|
| Fully connected		| outputs  320						|
| Fully connected		| outputs  160						|
| Fully connected		| outputs  80						|
| Fully connected		| outputs  40						|
| Fully connected		| outputs  10						|
| Fully connected		| outputs  1						|

#### 3. Creation of the Training Set & Training Process

In order to get good data, I drove with care to move the car to the center as much as possible. :D
In addition, the left and right inverted images were additionally learned based on the data obtained from the left and right center cameras.
Train data : Validation data = 8 : 2

Loss: MSE
Optimizer: Adam
Epoch: 3

The final model was saved in model.h5 and is the result of autonomous vehicle movement through drive.py.

[video1]
