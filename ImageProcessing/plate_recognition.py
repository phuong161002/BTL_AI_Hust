from ImageProcessing import detect_lp as detector
from ImageProcessing import preprocess
import cv2
import numpy as np
import string 
import random

def Recognize(img):
    # return list objects of plate: 
    #   grayimg = Gray Image of plate
    #   binaryimg = Binary Image of plate
    #   chars = list objects of character in plate:    
    #       {
    #           img = binary image of character
    #           rect = x,y,w,h of char in plate image
    #           name = random string to save image
    #       }

    plates, grayImg = detector.DetectWithHaarCascade(img)
    plateData = []
    for x, y, w, h in plates:
        plateImg = grayImg[y:y+h, x:x+w]
        plateImg = cv2.resize(plateImg, (400, 300))
        plate = {}
        plate['original'] = img[y:y+h, x:x+w]
        plate['grayimg'] = plateImg
        chars = []
        binaryImg = plateImg.copy()
        
        # cv2.threshold(plateImg, 50, 255, cv2.THRESH_BINARY_INV, binaryImg)
        cv2.adaptiveThreshold(plateImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 2.0, binaryImg)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
       
        cv2.dilate(binaryImg, kernel3, iterations=3)
        cv2.erode(binaryImg, kernel3, iterations=3)

        plate['binaryimg'] = binaryImg
        # thre_mor = cv2.morphologyEx(binaryImg, cv2.MORPH_DILATE, kerel3)
        contours, hier = cv2.findContours(binaryImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        listRect = []
        for contour in contours:
            x1,y1,w1,h1 = cv2.boundingRect(contour)
            
            if  30 < w1 < 150 and 80 < h1 < 150 and 0.2 < w1/h1 < 1.1:
                cropped = binaryImg[y1:y1+h1, x1:x1+w1]

                # remove roi is not a character by check ratio of white points
                if(cv2.countNonZero(cropped) / (w1 * h1) > 0.3):
                    charObj = {}
                    charObj['img'] = cropped
                    charObj['rect'] = (x1, y1, w1, h1)
                    charObj['name'] = randomString()
                    chars.append(charObj)
                    listRect.append((x1, y1, h1, w1))
                # cv2.imshow(str(x1)+str(y1)+str(w1)+str(h1), cropped)
            # cv2.drawContours(plateImg, [contour], 0, (0, 255, 0), 1) 
        plate['chars'] = chars

        plateData.append(plate)
    return plateData
  

def randomString():
    S = 10  # number of characters in the string.  
    # call random.choices() string module to find the string in Uppercase + numeric data.  
    ran = ''.join(random.choices(string.ascii_lowercase + string.digits, k = S))    
    return str(ran)