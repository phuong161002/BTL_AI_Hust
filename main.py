# import ImageProcessing.plate_recognition as PR
# import TextRecognition.text_recognition as TR
import ImageProcessing.preprocess as preprocess
import ImageProcessing.detect_lp as detector
import cv2
import os

projectDir = 'F:\\Python\\Car_Plate_Recognition\\BTL_AI_Hust'
imgDir = 'F:\\Python\\Car_Plate_Recognition\\BTL_AI_Hust\\Image\\Source'
imgFileName = 'test.jpg'
outputPlateImgDir = 'F:\\Python\\Car_Plate_Recognition\\BTL_AI_Hust\\Image\\Output\\Plate'

path = os.path.join(imgDir, imgFileName)

imgs = []


# Read all image in folder
for file in os.listdir(imgDir):
    if file.endswith('.jpg'):
        imgPath = os.path.join(imgDir, file)
        img = cv2.imread(imgPath)
        imgObj = {}
        imgObj['name'] = file
        imgObj['data'] = img
        imgs.append(imgObj)
# Loop through all image and recognize the plate license
for img in imgs:
    result = detector.DetectWithHaarCascade(imgOriginal=img['data'])

    for (x, y, w, h) in result:
        cropped = img['data'][y:y+h, x:x+w]
        outputPath = os.path.join(outputPlateImgDir, img['name'])
        
        cv2.imwrite(outputPath, cropped)

    print(len(result))

    # cv2.imshow('test img ' + str(img), img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# result = PR.Recognize(img)