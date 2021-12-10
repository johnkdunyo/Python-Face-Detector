import cv2
import random

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#img = cv2.imread('john.jpeg')
img = cv2.imread('sample2.jpg')



#convert image to gratscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cv2.imshow('jon', grayscaled_img)
#cv2.waitKey()


#next is to trained the algo.. but that has been done already

#detec face coordinates
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

print(face_coordinates)


#draw rectangles with the coordinates over the img


for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (random.randrange(256), random.randrange(256), random.randrange(256)), 4)


#cv2.rectangle(img, (148, 159), (148+227,159+ 227), (0, 255, 0), 2)

cv2.imshow('new image', img)
cv2.waitKey()


