from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from keras.optimizers import SGD
from imutils import paths
from cnn import LeNet
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of all images")
ap.add_argument("-m", "--model")
args = vars(ap.parse_args())

output_model=args["model"]

if(output_model==None):
    tt=time.localtime()
    timestr=time.strftime('%Y-%m-%d-%H-%M-%S', tt)
    output_model=str(timestr)+".hdf5"

print(output_model)
# initialize the list of data and labels
data = []
labels = []

size_width=64

# loop over the input images
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = imutils.resize(image, width=size_width)
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)
	#print(imagePath)
	#print("Label:"+label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

print("classTotals:"+str(classTotals)+",classWeight:"+str(classWeight)+"labels:"+str(len(classWeight)))
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
#random_state : int, RandomState instance or None, optional (default=None)
#If int, random_state is the seed used by the random number generator;
#If RandomState instance, random_state is the random number generator;
#If None, the random number generator is the RandomState instance used by np.random.
#
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.20, stratify=labels, random_state=42)

# initialize the model
print("[INFO] compiling model...")
#in this example there are only 2 type image in the database
model = LeNet.build(width=size_width, height=size_width, depth=1, classes=len(classWeight))
model.compile(loss="binary_crossentropy", optimizer="adam",
	metrics=["accuracy"])

#print("[INFO] compiling model...")
#model = LeNet.build(width=size_width, height=size_width, depth=1, classes=2)
#opt = SGD(lr=0.01)
#model.compile(loss="categorical_crossentropy", optimizer=opt,
#	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	class_weight=classWeight, batch_size=size_width, epochs=15, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=size_width)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save(output_model)
