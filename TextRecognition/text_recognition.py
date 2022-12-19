import numpy as np
import cv2


RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


# Upload KNN model 
npaClassifications = np.loadtxt("classificationS.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape(
    (npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train
kNearest = cv2.ml.KNearest_create()  # instantiate KNN object
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

def Char(binaryImg): 
    imgResized = cv2.resize(binaryImg, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
    npaResized = imgResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    npaResized = np.float32(npaResized)
    _, npaResults, neigh_resp, dists = kNearest.findNearest(npaResized, k = 3)
    strChar = str(chr(int(npaResults[0][0])))

    return strChar


