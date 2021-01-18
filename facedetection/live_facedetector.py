import cv2
import numpy as np
from tensorflow import keras

from facedetection import facedetector

model = keras.models.load_model('E:/Python/FaceMask/face-mask-detection/modeltraining/model-2/inceptionV3-model.h5')

if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)
    fd = facedetector.FaceDetector()

    while True:
        ret, frame = video_capture.read()
        face_frame = fd.get_face_frame(frame)

        if face_frame is not None:
            X_test = []

            roi = cv2.resize(face_frame, (100, 100))
            data = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            X_test.append(data)

            X_test = np.array(X_test)
            X_test = X_test / 255
            pic = np.array(X_test)

            Y_test = model.predict(pic)

            print("NO MASK" if np.argmax(Y_test, axis=1) == [0] else "MASK" if np.argmax(Y_test, axis=1) == [1] else "WRONG MASK")

            cv2.imshow('Video', fd.get_last_frame_live(Y_test))
        else:
            cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
