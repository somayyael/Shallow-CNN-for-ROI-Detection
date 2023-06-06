# -*- coding: utf-8 -*-
"""
Created on Mon May 31 01:36:42 2021

@author: selmo
"""

# import the necessary packages

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os
from skimage import util 
# determine the input file type, but assume that we're working with
# single input image
filetype = mimetypes.guess_type("test_images.txt")[0]
imagePaths = ["test_images.txt"]
# if the file type is a text file, then we need to process *multiple*
# images
if "text/plain" == filetype:
	# load the filenames in our testing file and initialize our list
	# of image paths
	filenames = open("test_images.txt").read().strip().split("\n")
	imagePaths = []
	# loop over the filenames
	for f in filenames:
		# construct the full path to the image filename and then
		# update our image paths list
		p = os.path.sep.join(["images", f])
		imagePaths.append(f)
        # load our trained bounding box regressor from disk
print("[INFO] loading object detector...")
model = load_model("output_new/detector_scnn.h5")
# loop over the images that we'll be testing using our bounding box
# regression model
i=0
for imagePath in imagePaths:
	# load the input image (in Keras format) from disk and preprocess
	# it, scaling the pixel intensities to the range [0, 1]
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image) / 255.0
	image = np.expand_dims(image, axis=0)
    	# make bounding box predictions on the input image
	preds = model.predict(image)[0]
	(startX, startY, endX, endY) = preds
	# load the input image (in OpenCV format), resize it such that it
	# fits on our screen, and grab its dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]
	# scale the predicted bounding box coordinates based on the image
	# dimensions
	startX = int(startX * w)
	startY = int(startY * h)
	endX = int(endX * w)
	endY = int(endY * h)
	# draw the predicted bounding box on the image
    
    # draw the predicted bounding box on the image
	cv2.rectangle(image, (startX, startY), (endX, endY),
		(0, 255, 0), 2)
	# show the output image
	# show the output image
	filename="output_new/"+str(i)+".png"
	cv2.imwrite(filename,image)
# Mask input image with binary mask

	mask = np.zeros(image.shape, dtype=np.uint8)
	mask=cv2.rectangle(mask, (startX, startY), (endX, endY),
		(255, 255, 255), -1)
	result = cv2.bitwise_and(image, mask)

	result[mask==0] = 255 # Optional
	filename="output_new/"+str(i)+"mask.png"
	mask=util.invert(mask)
	cv2.imwrite(filename,mask)
	i=i+1