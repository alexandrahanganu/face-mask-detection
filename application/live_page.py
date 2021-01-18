import sys

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QMessageBox, QFileDialog, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QTextEdit, \
    QApplication

from PyQt5.QtCore import QThread, Qt, pyqtSlot, QTimer, pyqtSignal
import cv2
import numpy as np
from tensorflow import keras

from facedetection import facedetector

model = keras.models.load_model('E:/Python/FaceMask/face-mask-detection/modeltraining/model-2/inceptionV3-model.h5')


class LivePage(QWidget):

    def __init__(self, parent=None):
        super(LivePage, self).__init__()
        self.padre = parent
        self.setupUI()

    def setupUI(self):
        self.VBL = QVBoxLayout(self)

        self.FeedLabel = QLabel(self)
        self.VBL.addWidget(self.FeedLabel)

        self.Worker1 = Worker1()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.setLayout(self.VBL)

    def ImageUpdateSlot(self, Image):
        """
        updates the image slot with current image
        """
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def closeEvent(self, event):
        """
        prompts a message box that states that the user is about to close live feed detection
        """
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.padre.toggle_buttons(True)
            event.accept()
            self.Worker1.stop()
            self.close()
        else:
            self.padre.test = False
            event.ignore()


class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    global model

    def run(self):
        """
        starts the thread which captures live feed and sends the data to the
        algorithm for further analyzing
        """
        self.ThreadActive = True
        video_capture = cv2.VideoCapture(0)
        fd = facedetector.FaceDetector()

        while self.ThreadActive:
            ret, frame = video_capture.read()

            if ret:
                face_frame = fd.get_face_frame(frame)
                if face_frame is not None:

                    X_test = []

                    roi = cv2.resize(face_frame, (100, 100))
                    data = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    X_test.append(data)

                    X_test = np.array(X_test)
                    X_test = X_test / 255.
                    pic = np.array(X_test)

                    try:
                        Y_test = model.predict(pic)
                    except Exception as e:
                        print(f'Exception: {e}')

                    live_frame = fd.get_last_frame_live(Y_test)
                    Image = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
                    ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0],
                                               QImage.Format_RGB888)
                    Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    self.ImageUpdate.emit(Pic)
                else:
                    Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0],
                                               QImage.Format_RGB888)
                    Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    self.ImageUpdate.emit(Pic)

    def stop(self):
        """
        stops the thread
        :return:
        """
        self.ThreadActive = False
        self.quit()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = LivePage()
    Root.show()
    sys.exit(App.exec())
