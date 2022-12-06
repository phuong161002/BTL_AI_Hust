import imutils
import numpy as np
import cv2
import cv2 as cv
from matplotlib import pyplot as plt
import pytesseract

lic_data = cv2.CascadeClassifier('./output-hv-33-x25.xml')


def plt_show(image, title="", gray=False, size=(100, 100)):
    temp = image
    if gray == False:
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        plt.title(title)
        plt.imshow(temp, cmap='gray')
        plt.show()


def detect_number(img):
    temp = img
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    number = lic_data.detectMultiScale(img, 1.2)
    print("number plate detected:"+str(len(number)))
    for numbers in number:
        (x, y, w, h) = numbers
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+h]
        cv2.rectangle(temp, (x, y), (x+w, y+h), (0, 255, 0), 3)

    plt_show(temp)


img = cv2.imread("./xemay15.jpg")
plt_show(img)
detect_number(img)
plt.subplot(1, 1, 1), plt.imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)

plt.subplot(1, 1, 1), plt.imshow(erosion)
plt.title(''), plt.xticks([]), plt.yticks([])
plt.show()


image = img
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)


img = cv2.resize(img, (620, 480))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
gray = cv2.bilateralFilter(gray, 13, 15, 15)

edged = cv2.Canny(gray, 30, 200)


contours = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

for c in contours:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break


if screenCnt is None:
    detected = 0
    print("No contour detected")
else:
    detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)


# Masking the part other than the number plate
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1,)
new_image = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite('./anhbienso/anh7.jpg', new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
