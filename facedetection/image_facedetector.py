import cv2
import numpy as np
from tensorflow import keras

from facedetection import facedetector

model = keras.models.load_model('E:/Python/FaceMask/face-mask-detection/modeltraining/model-2/inceptionV3-model.h5')

if __name__ == '__main__':
    fd = facedetector.FaceDetector()

    frame = cv2.imread("image2.bmp")

    faces_frame = fd.get_face_frame_mtcnn_image(frame)
    # print("Faces: ", faces_frame)

    if faces_frame is not None:
        Y_test = []

        for face in faces_frame:
            X_test = []

            roi = cv2.resize(face, (100, 100))
            data = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            X_test.append(data)

            X_test = np.array(X_test)
            X_test = X_test / 255
            pic = np.array(X_test)

            Y_test.append(model.predict(pic))

        cv2.imshow('Annotated picture', fd.get_last_frame_image(Y_test))
        cv2.waitKey(0)
    else:
        cv2.imshow('Picture', frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
