import ImageProcessing.plate_recognition as PR
import TextRecognition.text_recognition as TR
import cv2
import os
import config

def get_data(input_dir):
    # Return list image data of all images in input directory
    # image data :
    # {
    #   name : file name of image
    #   data : image
    #
    # }
    imgs = []
     # Read all image in folder
    for file in os.listdir(input_dir):
        if file.endswith('.jpg'):
            imgPath = os.path.join(config.imgDir, file)
            img = cv2.imread(imgPath)
            imgObj = {}
            imgObj['name'] = file
            imgObj['data'] = img
            imgs.append(imgObj)
        #end if
    #end for
    return imgs

def main():
    # Load data from input Directory
    imgs = get_data(config.imgDir)

    # Loop through all image and recognize the plate license
    for img in imgs:
        # Get info of this image : plate, characters in plate
        result = PR.Recognize(img['data'])
        for plate in result:
            # Save character image to output directory
            for char in plate['chars']:
                charImg = char['img']
                strChar = TR.Char(charImg)
                fileName = strChar + '-----' + char['name'] + '.jpg'
                charImgPath = os.path.join(config.outputDir, 'Chars', fileName)
                cv2.imwrite(charImgPath, charImg)
      
            # Save plate image to output directory
            outputPath = os.path.join(config.outputPlateImgDir, img['name'])
            cv2.imwrite(outputPath, plate['original'])



if __name__ == '__main__':
    main()