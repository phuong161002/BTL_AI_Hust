import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


img_dir = 'Images'


def get_data(input_path):
    dirs = os.listdir(input_path)

    training_set = []

    for dir in dirs:
        label_dir = input_path + '/' + dir
        imgs = os.listdir(label_dir)
        for img_file in imgs:
            img_path = label_dir + '/' + img_file
            img = cv2.imread(img_path, 0)
            data = {
                'filepath': img_path,
                'label':dir.title(),
                'data': np.array(img_processing(img))
            }
            training_set.append(data)
    return training_set


def img_processing(img):
    cv2.threshold(img, 1, 255, 0, img)
    return img



all_data = get_data('Images')

train_data = []
train_label = []

for data in all_data:
    train_data.append((data['data']))
    train_label.append(int(data['label']))


knn = cv2.ml.KNearest_create()
knn.train(np.reshape(train_data, (-1, 400)).astype(np.float32), 0, np.reshape(train_label, (-1, 1)))

testImg = cv2.imread('test.png', 0)
testImg = cv2.resize(testImg, (20, 20))

image = cv2.threshold(testImg, 1, 255, 0, testImg)
cv2.imshow('testimg',testImg)
cv2.waitKey(0)


test = np.array(testImg).reshape(-1, 400).astype(np.float32)

ret, result, neighbors, dist =  knn.findNearest(test, k=50)

print(ret)
print(result)
print(neighbors)
print(dist)
