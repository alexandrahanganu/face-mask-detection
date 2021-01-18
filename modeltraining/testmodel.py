import os

import cv2
from tensorflow import keras
import numpy as np

model = keras.models.load_model('E:/Python/FaceMask/face-mask-detection/modeltraining/model-3/inceptionV3-model.h5')

X_test = []

path = 'E:/Python/FaceMask/face-mask-detection/test-images'
for filename in os.listdir(path):
    img = cv2.imread(path + '/' + filename)
    roi = cv2.resize(img, (100, 100))
    data = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    X_test.append(data)
    print(filename)

X_test = np.array(X_test)
X_test = X_test/255.

Y_test = model.predict(X_test)

print(Y_test)
