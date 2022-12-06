import imutils
import numpy as np
import cv2
import cv2 as cv
from matplotlib import pyplot as plt


img = cv2.imread("./xemay2.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('input', gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
