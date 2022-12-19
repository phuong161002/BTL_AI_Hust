import ImageProcessing.plate_recognition as PR
import TextRecognition.text_recognition as TR
import ImageProcessing.preprocess as preprocess
import ImageProcessing.detect_lp as detector
import cv2
import os

projectDir = 'F:\\Python\\Car_Plate_Recognition\\BTL_AI_Hust'
imgDir = 'F:\\Python\\Car_Plate_Recognition\\BTL_AI_Hust\\Image\\Source'
outputPlateImgDir = 'F:\\Python\\Car_Plate_Recognition\\BTL_AI_Hust\\Image\\Output\\Plate'

def get_data(input_dir):
    imgs = []
     # Read all image in folder
    for file in os.listdir(input_dir):
        if file.endswith('.jpg'):
            imgPath = os.path.join(imgDir, file)
            img = cv2.imread(imgPath)
            imgObj = {}
            imgObj['name'] = file
            imgObj['data'] = img
            imgs.append(imgObj)
        #end if
    #end for
    return imgs

def main_():
    imgs = get_data(imgDir)
    plate = detector.Detect(imgs[0]['data'])
    print(plate)


def main():
    imgs = get_data(imgDir)

    # Loop through all image and recognize the plate license
    for img in imgs:
        # result = detector.DetectWithHaarCascade(imgOriginal=img['data'])
        result = PR.Recognize(img['data'])
        for plate in result:
            for char in plate['chars']:
                cv2.imshow(str(char['rect']), char['img'])
            cv2.imshow('grayimg', plate['grayimg'])
            cv2.imshow('binaryimg', plate['binaryimg'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        # for (x, y, w, h) in result:
        #     cropped = img['data'][y:y+h, x:x+w]
        #     outputPath = os.path.join(outputPlateImgDir, img['name'])
            
        #     cv2.imwrite(outputPath, cropped)

        print(len(result))

        # cv2.imshow('test img ' + str(img), img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # result = PR.Recognize(img)            


if __name__ == '__main__':
    main()