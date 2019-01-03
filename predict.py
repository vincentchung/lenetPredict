from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import contours
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

size_width=64

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-m", "--model", required=True,
	help="path to input model")
args = vars(ap.parse_args())

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

imagePaths = list(paths.list_images(args["input"]))

for imagePath in imagePaths:
	# load the image and convert it to grayscale, then pad the image
	# to ensure digits caught only the border of the image are
	# retained
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (size_width, size_width))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	(pred1, pred2) = model.predict(image)[0]
	label = "chef" if pred1 > pred2 else "doctor"
	print(imagePath+":"+label+":"+str(pred1)+":"+str(pred2))
    #print(str(pred1))

	# show the output image
