import cv2
import numpy as np
import mtcnn


class FaceDetector:

    def __init__(self):
        self.cascadePath = "E:/Python/FaceMask/face-mask-detection/facedetection/haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath)
        self.last_frame = None
        self.last_face = None
        self.detector = mtcnn.MTCNN()
        self.faces = None

    def get_last_frame_live(self, label):
        """
        The function uses the last frame from the class in order to highlight the identified largest
        (closest) face and label.
        Args:
            label (numpy array): label for the face
        Returns:
            (numpy array): last frame labeled
        """
        if self.last_face is not None:
            cv2.rectangle(self.last_frame, (self.last_face[0], self.last_face[1]),
                          (self.last_face[0] + self.last_face[2], self.last_face[1] + self.last_face[3]),
                          (0, 255, 0) if np.argmax(label, axis=1) == [1] else
                          (0, 0, 255) if np.argmax(label, axis=1) == [0] else (0, 255, 255),
                          2)
            cv2.putText(self.last_frame,
                        "NO MASK" if np.argmax(label, axis=1) == [0] else
                        "MASK" if np.argmax(label, axis=1) == [1] else "WRONG MASK",
                        (self.last_face[0], self.last_face[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if np.argmax(label, axis=1) == [1] else
                        (0, 0, 255) if np.argmax(label, axis=1) == [0] else (0, 255, 255), 2)
        return self.last_frame

    def get_face_frame(self, frame):
        """
        The function uses get_face_frame_opencv in order to deliver the largest face from the frame.
        Args:
            frame (numpy array): entire frame received from the camera
        Returns:
            (numpy array): largest face
            (None): if the function doesn't find any face
        """
        self.last_frame = frame
        face_frame = self.get_face_frame_opencv(frame)
        if face_frame is not None:
            self.last_face = face_frame
            return frame[face_frame[1]:face_frame[1] + face_frame[3], face_frame[0]:face_frame[0] + face_frame[2]]
        else:
            return None

    def get_face_frame_opencv(self, frame):
        """
        The function returns the largest (closest) face using opencv.
        Args:
            frame (numpy array): entire frame received from the camera
        Returns:
            (numpy array): largest face
            (None): if the function doesn't find any face
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        except Exception as e:
            print(f'Exception: {e}')

        if len(faces) > 0:
            if len(faces) == 1:
                return faces[0][0], faces[0][1], faces[0][2], faces[0][3]
            else:
                largest_face = faces[0]
                for face in faces[1:]:
                    if largest_face[2] * largest_face[3] < face[2] * face[3]:
                        largest_face = face
                return largest_face[0], largest_face[1], largest_face[2], largest_face[3]
        else:
            return None

    def get_face_frame_mtcnn(self, frame):
        """
        The function returns the faces identified using mtcnn.
        Args:
           frame (numpy array): entire frame received from the camera
        Returns:
           (numpy array): faces
           (None): if the function doesn't find any face
        """
        faces = self.detector.detect_faces(frame)
        if len(faces):
            faces_list = []
            for face in faces:
                x, y, width, height = face['box']
                faces_list.append([x, y, width, height])
            return faces_list
        else:
            return None

    def get_face_frame_mtcnn_image(self, frame):
        """
        The function returns the faces using mtcnn.
        Args:
            frame (numpy array): entire frame received from the camera
        Returns:
            (numpy array): largest face
            (None): if the function doesn't find any face
        """
        self.last_frame = frame
        faces_frame = self.get_face_frame_mtcnn(frame)
        self.faces = faces_frame
        if faces_frame is not None:
            face_list = []
            for face in faces_frame:
                face_list.append(frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]])
            return face_list
        else:
            return None

    def get_last_frame_image(self, labels):
        """
        The function uses the last frame from the class in order to highlight the identified faces and labels.
        Args:
            labels (numpy array): labels for the faces
        Returns:
            (numpy array): last frame labeled
        """
        if self.faces is not None:
            for i, face in enumerate(self.faces):
                cv2.rectangle(self.last_frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]),
                              (0, 255, 0) if np.argmax(labels[i], axis=1) == [1] else
                              (0, 0, 255) if np.argmax(labels[i], axis=1) == [0] else (0, 255, 255),
                              2)
                cv2.putText(self.last_frame,
                            "NO MASK" if np.argmax(labels[i], axis=1) == [0] else
                            "MASK" if np.argmax(labels[i], axis=1) == [1] else "WRONG MASK",
                            (face[0], face[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.005*face[2],
                            (0, 255, 0) if np.argmax(labels[i], axis=1) == [1] else
                            (0, 0, 255) if np.argmax(labels[i], axis=1) == [0] else (0, 255, 255), 2)
        return self.last_frame
