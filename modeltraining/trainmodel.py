import os
import pickle

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.utils import to_categorical

model_number = 11


def save_arrays():
    """
    saves the X and Y arrays for the current dataset and returns them afterwards
    """
    label2category = {'without_mask': 0, 'with_mask': 1, 'wrong_mask': 2}
    category2label = {v: k for k, v in label2category.items()}
    path = 'E:/Python/FaceMask/Balanced'

    X = []
    Y = []

    i = 0

    dirs = [path + '/without_mask', path + '/with_mask', path + '/wrong_mask']
    for directory in dirs:
        if directory == path + '/without_mask':
            cat = 0
        elif directory == path + '/with_mask':
            cat = 1
        else:
            cat = 2

        for filename in os.listdir(directory):
            img = cv2.imread(directory + '/' + filename)
            i = i + 1
            roi = cv2.resize(img, (100, 100))
            data = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            target = to_categorical(cat, num_classes=len(category2label))
            X.append(data)
            Y.append(target)

    X = np.array(X)
    Y = np.array(Y)

    X = X / 255.

    os.mkdir(f"E:/Python/FaceMask/face-mask-detection/modeltraining/model-{model_number}")

    file = open(f"E:/Python/FaceMask/face-mask-detection/modeltraining/model-{model_number}/X.txt", "wb")
    pickle.dump(X, file)
    file.close()

    file = open(f"E:/Python/FaceMask/face-mask-detection/modeltraining/model-{model_number}/Y.txt", "wb")
    pickle.dump(Y, file)
    file.close()

    return X, Y


def load_arrays():
    """
    loads the saved X and Y arrays for the current dataset and returns them afterwards
    """

    # os.mkdir(f"E:/Python/FaceMask/face-mask-detection/modeltraining/model-{model_number}")

    file = open(f"E:/Python/FaceMask/face-mask-detection/modeltraining/model-7/X.txt", "rb")
    X = pickle.load(file)

    file = open(f"E:/Python/FaceMask/face-mask-detection/modeltraining/model-7/Y.txt", "rb")
    Y = pickle.load(file)

    return X, Y


def trainmodel(saved=True):
    """
    trains the model with the given data, prints the evaluation of the model and saves the model
    """
    if saved is True:
        X, Y = load_arrays()
    else:
        X, Y = save_arrays()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    pre_trained_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(100, 100, 3))

    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_layer.output

    x = tf.keras.layers.Flatten()(last_output)
    x = tf.keras.layers.Dense(1024, activation='swish')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=pre_trained_model.input, outputs=x)
    model.compile(optimizer='adam', loss='kullback_leibler_divergence', metrics=['acc'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='data/model-{epoch:03d}.ckpt', save_weights_only=True,
                                                    monitor='val_acc', mode='max', save_best_only=True, verbose=0)

    model.fit(X_train, Y_train, epochs=20, callbacks=[checkpoint], validation_split=0.1)
    model.save(f'E:/Python/FaceMask/face-mask-detection/modeltraining/model-{model_number}/inceptionV3-model.h5')

    print(model.evaluate(X_test, Y_test))


if __name__ == "__main__":
    trainmodel(saved=True)
