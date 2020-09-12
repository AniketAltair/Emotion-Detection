import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import warnings
from keras.models import model_from_json
import os


img=cv2.imread(r"C:/Users/Aniketh/Desktop/Emodet/Emotions/train/angry/PrivateTest_88305.jpg")
print(img.shape)
input_shape=(48,48,3)

from keras.preprocessing.image import ImageDataGenerator

image_gen=ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             rescale=1/255,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')


from keras.models import Sequential
from keras.layers import Activation,Dropout,Flatten,Conv2D,MaxPooling2D,Dense

model=Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(48,48,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(48,48,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(48,48,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(7,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
print(model.summary())

batch_size=15
train_image_gen=image_gen.flow_from_directory(r"C:/Users/Aniketh/Desktop/Emodet/emotions_new/train",
                                              target_size=input_shape[:2],
                                              batch_size=batch_size,
                                              class_mode='categorical')

test_image_gen=image_gen.flow_from_directory(r"C:/Users/Aniketh/Desktop/Emodet/emotions_new/test",
                                              target_size=input_shape[:2],
                                              batch_size=batch_size,
                                              class_mode='categorical')

print(train_image_gen.class_indices)

results=model.fit(train_image_gen,epochs=15,steps_per_epoch=1000,validation_data=test_image_gen,validation_steps=12)

warnings.filterwarnings('ignore')

print(results.history['accuracy'])

model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model1.h5")
print("Saved model to disk")


json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")

from keras.preprocessing import image

test_img=r"C:\Users\Aniketh\Desktop\Emodet\angry.jpg"
test_img=image.load_img(test_img,target_size=(48,48))
test_img=image.img_to_array(test_img)
test_img=np.expand_dims(test_img,axis=0)# adds 1 to dimension=> (48,48,3) becomes (1,48,48,3)
test_img=test_img/255
print(loaded_model.predict_classes(test_img))

test_img=r"C:\Users\Aniketh\Desktop\Emodet\sad.jpg"
test_img=image.load_img(test_img,target_size=(48,48))
test_img=image.img_to_array(test_img)
test_img=np.expand_dims(test_img,axis=0)# adds 1 to dimension=> (48,48,3) becomes (1,48,48,3)
test_img=test_img/255
print(loaded_model.predict_classes(test_img))

test_img=r"C:\Users\Aniketh\Desktop\Emodet\happy.jpg"
test_img=image.load_img(test_img,target_size=(48,48))
test_img=image.img_to_array(test_img)
test_img=np.expand_dims(test_img,axis=0)# adds 1 to dimension=> (48,48,3) becomes (1,48,48,3)
test_img=test_img/255
print(loaded_model.predict_classes(test_img))




