# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 22:43:06 2021

@author: selmo
"""

import numpy as np
import os,sys
# os.environ['THEANO_FLAGS'] = "device=gpu0,floatX=float32"
import time
#import cPickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import linalg as LA
from sklearn.metrics.pairwise import euclidean_distances
#import hickle as hkl
from tensorflow.keras.optimizers import SGD, Adam, Adadelta
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Reshape, Activation
#from keras.utils.visualize_util import plot
#from keras.layers.core import Layer
#from keras.regularizers import l2
#from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
#from tensorflow.keras.callbacks import ModelCheckpoint

#from tensorflow.keras.layers import Flatten


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import cv2


def CNN_model(img_rows, img_cols, channel=3, num_class=None):

    input = Input(shape=(channel, img_rows, img_cols))
    conv1_3x3= Convolution2D(32,3,3,name='conv1/7x7',activation='relu')(input)#,W_regularizer=l2(0.0002)
    conv2_3x3= Convolution2D(32,3,3,name='conv2/3x3',activation='relu')(conv1_3x3)
    pool2_2x2= MaxPooling2D(pool_size=(2,2),strides=(1,1),border_mode='valid',name='pool2')(conv2_3x3)
    
    conv3_3x3= Convolution2D(32,3,3,name='conv1/7x7',activation='relu')(pool2_2x2)#,W_regularizer=l2(0.0002)
    conv4_3x3= Convolution2D(32,3,3,name='conv2/3x3',activation='relu')(conv3_3x3)
    pool3_2x2= MaxPooling2D(pool_size=(2,2),strides=(1,1),border_mode='valid',name='pool3')(conv4_3x3)
    
    poll_flat = Flatten()(pool3_2x2)
    #fully connected layer
    
    
    #MLP
    fc_1 = Dense(512,name='fc_1',activation='relu')(poll_flat)
    drop_fc = Dropout(0.5)(fc_1)
    out = Dense(240,name='fc_2',activation='sigmoid')(drop_fc)
    # Create model
    model = Model(input=input, output=out)
    # Load cnn pre-trained data 
    #model.load_weights('models/weights.h5')#NOTE 
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  #  adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=Adadelta, loss='mean_absolute_error')  
    return model

#cap = cv2.VideoCapture('Open_appendectomy.mp4')

#image = cv2.imread(input)
#(h, w) = image.shape[:2]
#blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

print("[INFO] loading dataset...")
rows = open("output.csv").read().strip().split("\n")
# initialize the list of data (images), our target output predictions
# (bounding box coordinates), along with the filenames of the
# individual images
data = []
targets = []
filenames = []
# loop over the rows
for row in rows:
	# break the row into the filename and bounding box coordinates
	row = row.split(",")
	(file, startX, startY, endX, endY) = row
    	# derive the path to the input image, load the image (in OpenCV
	# format), and grab its dimensions
	file=file.split("/")
	filename=file[1].split("___")
	filename="images/"+filename[0]
	#filename=filename[0]
	imagePath = os.path.sep.join(["images", filename])
	print(filename)
	image = cv2.imread(filename)
	(h, w) = image.shape[:2]
	# scale the bounding box coordinates relative to the spatial
	# dimensions of the input image
	startX = float(startX) / w
	startY = float(startY) / h
	endX = float(endX) / w
	endY = float(endY) / h
    	# load the image and preprocess it
	image = load_img(filename, target_size=(224, 224))
	image = img_to_array(image)
	# update our list of data, targets, and filenames
	data.append(image)
	targets.append((startX, startY, endX, endY))
   
	filenames.append(filename)
    
# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")
# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing
split = train_test_split(data, targets, filenames, test_size=0.10,
	random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]
# write the testing filenames to disk so that we can use then
# when evaluating/testing our bounding box regressor
print("[INFO] saving testing filenames...")
f = open("test_images1.txt", "w")
f.write("\n".join(testFilenames))
f.close()


#Data Augmentation
trainAug = ImageDataGenerator(
	rotation_range=180,
	zoom_range=0.25,
	width_shift_range=0.25,
	height_shift_range=0.25,
	shear_range=0.2,
	rescale=1/255,
	fill_mode="nearest")



model = CNN_model(244,244)
print(model.summary())
# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=32,
	epochs=25,
	verbose=1)
# serialize the model to disk
print("[INFO] saving object detector model...")
model.save("output/detectorscnn.h5", save_format="h5")
N = 25
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("plot1.png")

print("[INFO] saved...")