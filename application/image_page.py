from os.path import expanduser

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QGraphicsDropShadowEffect
from tensorflow import keras

from application.controller import PageWindow
from facedetection import facedetector


class ImagePage(PageWindow):

    def __init__(self):
        super().__init__()
        self.ok = 1
        self.setupUi()

    def setupUi(self):
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(15)

        self.toolButton = QtWidgets.QToolButton(self)
        self.toolButton.setGeometry(QtCore.QRect(850, 80, 40, 20))
        self.toolButton.setObjectName("toolButton")
        self.toolButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton.clicked.connect(self.browse_file_event)

        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setGeometry(QtCore.QRect(970, 100, 140, 60))
        self.pushButton.setMinimumSize(QtCore.QSize(100, 50))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.clicked.connect(self.save_event)
        self.pushButton.setDisabled(True)

        self.path = QtWidgets.QLineEdit(self)
        self.path.setGeometry(QtCore.QRect(270, 70, 560, 40))
        self.path.setObjectName("path")
        self.path.setDisabled(True)
        self.path.setPlaceholderText("Path to the file to analyze...")

        self.path_1 = QtWidgets.QLineEdit(self)
        self.path_1.setGeometry(QtCore.QRect(270, 150, 560, 40))
        self.path_1.setObjectName("path_1")
        self.path_1.setDisabled(True)
        self.path_1.setPlaceholderText("Path to the directory to save...")

        self.toolButton_1 = QtWidgets.QToolButton(self)
        self.toolButton_1.setGeometry(QtCore.QRect(850, 160, 40, 20))
        self.toolButton_1.setObjectName("toolButton_1")
        self.toolButton_1.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton_1.clicked.connect(self.browse_directory_event)

        self.pushButton_1 = QtWidgets.QPushButton(self)
        self.pushButton_1.setGeometry(QtCore.QRect(1170, 730, 100, 50))
        self.pushButton_1.setMinimumSize(QtCore.QSize(100, 50))
        self.pushButton_1.setObjectName("pushButton_1")
        self.pushButton_1.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_1.clicked.connect(self.back_event)

        self.pushButton_2 = QtWidgets.QPushButton(self)
        self.pushButton_2.setGeometry(QtCore.QRect(1170, 650, 100, 50))
        self.pushButton_2.setMinimumSize(QtCore.QSize(100, 50))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_2.clicked.connect(self.show_event)

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(100, 230, 1020, 530))
        self.label.setObjectName("label")
        self.label.setStyleSheet("background-color: rgb(220,220,220); border: 1px solid black;")
        self.label.setGraphicsEffect(self.shadow)

        self.label_1 = QtWidgets.QLabel(self)
        self.label_1.setGeometry(QtCore.QRect(300, 740, 600, 20))
        self.label_1.setObjectName("label_1")
        self.label_1.setText("")
        self.label_1.setAlignment(QtCore.Qt.AlignCenter)

        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.toolButton.setText(_translate("Dialog", "..."))
        self.pushButton.setText(_translate("Dialog", "Save"))
        self.toolButton_1.setText(_translate("Dialog", "..."))
        self.pushButton_1.setText(_translate("Dialog", "Back"))
        self.pushButton_2.setText(_translate("Dialog", "Status"))

    def browse_file_event(self):
        """
        opens browser to choose path to image file
        """
        self.path.setText("")
        # opens a file dialog for the format
        file_path = QFileDialog.getOpenFileName(None, 'Select a file:', expanduser("~"),
                                                filter="All files (*.*);;BMP (*.bmp);;CUR (*.cur);;GIF (*.gif);;ICNS "
                                                       "(*.icns);;ICO (*.ico);;JPEG (*.jpeg);;JPG (*.jpg);;PBM ("
                                                       "*.pbm);;PGM (*.pgm);;PNG (*.png);;PPM (*.ppm);;SVG ("
                                                       "*.svg);;SVGZ (*.svgz);;TGA (*.tga);;TIF (*.tif);;TIFF ("
                                                       "*.tiff);;WBMP (*.wbmp);;WEBP (*.webp);;XBM (*.xbm);;XPM ("
                                                       "*.xpm)")
        self.path.setText(file_path[0])
        self.check_for_save()

    def browse_directory_event(self):
        """
        opens browser to choose path to folder to save image
        """
        self.path_1.setText("")
        directory_path = QFileDialog.getExistingDirectory(None, 'Select a folder:', expanduser("~"))
        self.path_1.setText(directory_path)
        self.check_for_save()

    def check_for_save(self):
        """
        enables/disables the save button depending on whether the paths are filled or not
        """
        if self.path_1.text() != "" and self.path.text() != "":
            self.pushButton.setDisabled(False)
        else:
            self.pushButton.setDisabled(True)

    def save_event(self):
        """
        gets the image from the path, applies the algorithm and then saves the annotated picture
        """
        model = keras.models.load_model(
            'E:/Python/FaceMask/face-mask-detection/modeltraining/model-2/inceptionV3-model.h5')

        fd = facedetector.FaceDetector()

        frame = cv2.imread(self.path.text())

        faces_frame = fd.get_face_frame_mtcnn_image(frame)

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

            cv2.imwrite(self.path_1.text() + '/image.png', fd.get_last_frame_image(Y_test))
            self.label.setStyleSheet(f"border-image: url({self.path_1.text() + '/image.png'}); ")
            self.label_1.setStyleSheet("color: rgb(34,139,34); background-color: rgb(220,220,220);")
            self.label_1.setText(f"Image saved successfully at: {self.path_1.text() + '/image.png'}")
        else:
            self.label.setStyleSheet(f"border-image: url({self.path.text()});")
            self.label_1.setStyleSheet("color: rgb(178,34,34); background-color: rgb(220,220,220);")
            self.label_1.setText(f"No faces detected on image from: {self.path.text()}")

        cv2.destroyAllWindows()

    def clear_all(self):
        """
        clears all of the selections on pressing back
        """
        self.path_1.setText("")
        self.path.setText("")
        self.pushButton.setDisabled(True)
        self.label.setStyleSheet("background-color: rgb(220,220,220); border: 1px solid black;")
        self.label_1.setText("")
        self.label_1.setGraphicsEffect(None)

    def back_event(self):
        """
        on clicking the linked button, it opens the corresponding page and call for clearing the selections
        """
        self.goto('main')
        self.clear_all()

    def show_event(self):
        """
        on clicking the linked button, it shows or hides the label for status
        """
        self.ok = 1 - self.ok
        self.label_1.setHidden(False if self.ok == 1 else True)
