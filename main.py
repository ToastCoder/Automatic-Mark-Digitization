# IMPORTING THE ESSENTIAL LIBRARIES
import cv2
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image

# GETTING THE INPUT IMAGE AND PREVIEWING IT
raw = cv2.imread('data/raw_image.jpg',3)
raw = cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
plt.imshow(raw)
plt.title("Input Image")
plt.show()

# CROPPING THE MARK WORDS FROM THE IMAGE
y1=1836
x1=2000
h1=240
w1=600
crop_mark = raw[y1:y1+h1, x1:x1+w1]
crop_mark = cv2.cvtColor(crop_mark,cv2.COLOR_BGR2RGB)
plt.imshow(crop_mark)
plt.title("Image of marks field")
plt.show()

# CROPPING THE REG.NO FROM THE IMAGE
x2 = 1935
y2 = 835
w2 = 953
h2 = 100
crop_roll_no = raw[y2:y2+h2, x2:x2+w2]
plt.imshow(crop_roll_no)
plt.title("Image of Reg.No field")
plt.show()

# CONVERTING MARK IMAGE INTO GRAYSCALE
gray_mark = cv2.cvtColor(crop_mark,cv2.COLOR_BGR2GRAY)
gray_mark = cv2.cvtColor(gray_mark,cv2.COLOR_BGR2RGB)
plt.imshow(gray_mark)
plt.title("Mark Image in Grayscale")
plt.show()

# CONVERTING REG.NO IMAGE INTO GRAYSCALE
gray_roll_no = cv2.cvtColor(crop_roll_no,cv2.COLOR_BGR2GRAY)
gray_roll_no = cv2.cvtColor(gray_roll_no,cv2.COLOR_BGR2RGB)
plt.imshow(gray_roll_no)
plt.title("Reg.No Image into Grayscale")
plt.show()

# BINARIZATION OF MARK IMAGE
t , binary_mark = cv2.threshold(gray_mark,160,255,cv2.THRESH_BINARY)
binary_mark = cv2.cvtColor(binary_mark,cv2.COLOR_BGR2RGB)
plt.imshow(binary_mark)
plt.title("Binarization of Mark Image")
plt.show()

# BINARIZATION OF REG.NO IMAGE
t2 , binary_roll_no = cv2.threshold(gray_roll_no,160,255,cv2.THRESH_BINARY)
binary_roll_no = cv2.cvtColor(binary_roll_no,cv2.COLOR_BGR2RGB)
plt.imshow(binary_roll_no)
plt.title("Binarization of Reg.no Image")
plt.show()

# CONVERTING IMAGE INTO STRING AND PRINTING IT
print(pytesseract.image_to_string(binary_mark))
print(pytesseract.image_to_string(binary_roll_no))
