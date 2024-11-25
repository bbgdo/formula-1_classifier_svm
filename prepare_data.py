import os
import numpy as np
import cv2 as cv
import pickle

# https://www.kaggle.com/datasets/loveymishra/f1-image-classification-updated

DIR = 'data'
categories = ['Mercedes', 'Red Bull']
data = []
IMG_SIZE = 256

for category in categories:
    path = os.path.join(DIR, category)
    label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        car_img = cv.imread(img_path, 1)
        try:
            resized_img = cv.resize(car_img, (IMG_SIZE, IMG_SIZE))
            image = np.array(resized_img).flatten()
            data.append([image, label])
        except Exception as e:
            pass

print(len(data))

pick_in = open('f1_data.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()