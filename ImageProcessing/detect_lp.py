import cv2
import imutils
import numpy as np
from ImageProcessing import preprocess

# Param
max_size = 5000
min_size = 900

cascadePath = '.\\Model\\HaarCascade\\plate_cascade.xml'
plateCascade = cv2.CascadeClassifier(cascadePath)

def DetectWithHaarCascade(imgOriginal): 
    gray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    gray = preprocess.maximizeContrast(gray)
    plates:tuple = plateCascade.detectMultiScale(gray, 1.1, 8, 1, (25, 25))
    # for (x, y, w, h) in plates:
    #     cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #     plate = gray[y: y+h, x:x+w]
    #     plate = cv2.blur(plate, ksize=(20,20))
    #     gray[y:y+h, x:x+w] = plate

    return plates, gray #list of (x, y, w, h)
