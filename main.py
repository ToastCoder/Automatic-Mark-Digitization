import cv2
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image

image = cv2.imread('data/raw_image.jpg',3)

y=1836
x=2000
h=240
w=600
crop = image[y:y+h, x:x+w]

gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)

t , binary = cv2.threshold(gray,160,255,cv2.THRESH_BINARY)

cv2.imwrite('crop.jpg',binary)
