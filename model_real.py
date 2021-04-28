import cv2
import csv
import numpy as np
import os
import sklearn
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

def model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    
    model.add(Convolution2D(16,3,3, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(32,3,3, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,3,3, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(128,3,3, activation='relu'))
    model.add(Convolution2D(128,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(320))
    model.add(Dense(160))
    model.add(Dense(80))
    model.add(Dense(40))
    model.add(Dense(20))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def generator(data):
    num_data = len(data)
    bs = 64
    while 1: 
        samples = sklearn.utils.shuffle(data)
        for offset in range(0, num_data, bs):
            minibatch = data[offset:offset+bs]
            images = []
            measurements = []
            for image, measurement in minibatch:
                images.append(image)
                measurements.append(measurement)

            input = np.array(images)
            label = np.array(measurements)
            yield sklearn.utils.shuffle(inputs, label)
            
lines = []
with open('/root/Desktop/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    source_path_center = line[0]
    source_path_left = line[1]
    source_path_right = line[2]
    source_path_measurement = line[3]
    filename_center = source_path_center.split('/')[-1]
    filename_left = source_path_left.split('/')[-1]
    filename_right = source_path_right.split('/')[-1]
    filename_measurement = source_path_measurement
    current_path_center = '/root/Desktop/data/IMG/' + filename_center
    current_path_left = '/root/Desktop/data/IMG/' + filename_left
    current_path_right = '/root/Desktop/data/IMG/' + filename_right
    current_path_measurement = filename_measurement
    
    image_center = cv2.imread(current_path_center)
    image_left = cv2.imread(current_path_left)
    image_right = cv2.imread(current_path_right)
    measurement = float(current_path_measurement)
    images.append(image_center)
    measurements.append(measurement)
    images.append(image_left)
    measurements.append(measurement+0.2)
    images.append(image_right)
    measurements.append(measurement-0.2)
    images.append(cv2.flip(image_center,1))
    measurements.append(measurement*-1.0)

data = list(zip(images, measurements))
# print('Total samples: {}'.format( samples))
train_data, val_data = train_test_split(data, test_size=0.2)

train_gen = generator(train_data)
val_gen = generator(val_data)

model = model()
model.compile(loss='mse', optimizer='adam')
result = model.fit_generator(train_gen, samples_per_epoch= len(train_data), validation_data=val_gen, nb_val_samples=len(val_data), nb_epoch=3, verbose=1)
model.save('model.h5')
