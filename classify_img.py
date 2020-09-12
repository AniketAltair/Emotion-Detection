import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import warnings
from keras.models import model_from_json
import os
from keras.preprocessing import image


def imgprocess(img):
        json_file=open('model1.json','r')
        loaded_model_json=json_file.read()
        json_file.close()
        loaded_model=model_from_json(loaded_model_json)
        loaded_model.load_weights("model1.h5")
        return loaded_model.predict_classes(img)

def show(img,face_rects):
        for(x,y,w,h) in face_rects:
                i=img[y:y+h,x:x+w]
                i=cv2.resize(i,(48,48), interpolation = cv2.INTER_AREA)
                return i
               
