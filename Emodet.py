import cv2
import matplotlib.pyplot as plt 
import numpy as np 
import classify_image
from PIL import Image
import warnings
from keras.models import model_from_json
import os
from keras.preprocessing import image

cap=cv2.VideoCapture(0)

def det_face(img):
  face_img=img.copy()
  face_cascade=cv2.CascadeClassifier(r'C:\Users\Aniketh\Desktop\Emodet\haarcascade_frontalface_alt.xml')
  face_rects=face_cascade.detectMultiScale(face_img)
  for(x,y,w,h) in face_rects:
                cv2.rectangle(face_img,(x,y),(x+w,y+h),(0,255,0),5)
                i=classify_image.show(img,face_rects)
                test_img=image.img_to_array(i)
                test_img=np.expand_dims(test_img,axis=0)
                test_img=test_img/255
                print(classify_image.imgprocess(test_img))
                
        

  return face_img

while True:
  ret,frame=cap.read()
  frame=det_face(frame)
  cv2.imshow('Video Face Detect',frame)

  if cv2.waitKey(1)==27:
    break

cv2.release()
cv2.destroyAllWindows()

