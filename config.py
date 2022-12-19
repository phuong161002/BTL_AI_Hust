import os

projectDir = 'F:\\Python\\Car_Plate_Recognition\\BTL_AI_Hust'
imgDir = os.path.join('Image\\Source')
outputDir = os.path.join('Image\\Output')
outputPlateImgDir = os.path.join('Image\\Output\\Plate')
cascadePath = os.path.join(projectDir, 'Model\\HaarCascade\\plate_cascade.xml') 