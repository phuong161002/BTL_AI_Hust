from ImageProcessing import detect_lp as detector
from ImageProcessing import preprocess
import cv2
import numpy as np

def Recognize(img):
    plates, grayImg = detector.DetectWithHaarCascade(img)
    plateData = []
    for x, y, w, h in plates:
        plateImg = grayImg[y:y+h, x:x+w]
        plateImg = cv2.resize(plateImg, (400, 300))
        plate = {}
        plate['grayimg'] = plateImg
        chars = []
        binaryImg = plateImg.copy()
        
        # cv2.threshold(plateImg, 50, 255, cv2.THRESH_BINARY_INV, binaryImg)
        cv2.adaptiveThreshold(plateImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 2.0, binaryImg)
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
       
        cv2.dilate(binaryImg, kerel3, iterations=3)
        cv2.erode(binaryImg, kerel3, iterations=3)

        plate['binaryimg'] = binaryImg
        # thre_mor = cv2.morphologyEx(binaryImg, cv2.MORPH_DILATE, kerel3)
        contours, hier = cv2.findContours(binaryImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x1,y1,w1,h1 = cv2.boundingRect(contour)
            if  30 < w1 < 150 and 80 < h1 < 150 and 0.2 < w1/h1 < 1.1:
                cropped = binaryImg[y1:y1+h1, x1:x1+w1]
                charObj = {}
                charObj['img'] = cropped
                charObj['rect'] = (x1, y1, w1, h1)
                chars.append(charObj)
                # cv2.imshow(str(x1)+str(y1)+str(w1)+str(h1), cropped)
            # cv2.drawContours(plateImg, [contour], 0, (0, 255, 0), 1) 
        plate['chars'] = chars

        plateData.append(plate)
        # cv2.imshow('img', plateImg)
        # cv2.imshow('img2', binaryImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # CharSeparator(plateImg) #Origin Plate
    return plateData

def CharSeparator(imgPlateLicense):
    # gray = cv2.cvtColor(imgPlateLicense, cv2.COLOR_BGR2GRAY)
    img = preprocess.maximizeContrast(imgPlateLicense)
    return img
    
