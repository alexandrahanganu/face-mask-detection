from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGraphicsDropShadowEffect

from application.controller import PageWindow


class AboutPage(PageWindow):

    def __init__(self):
        super().__init__()
        self.ok = 1
        self.setupUi()

    def setupUi(self):
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(15)

        self.pushButton_1 = QtWidgets.QPushButton(self)
        self.pushButton_1.setGeometry(QtCore.QRect(1170, 730, 100, 50))
        self.pushButton_1.setMinimumSize(QtCore.QSize(100, 50))
        self.pushButton_1.setObjectName("pushButton_1")
        self.pushButton_1.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_1.clicked.connect(self.back_event)

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(100, 230, 1020, 530))
        self.label.setObjectName("label")
        self.label.setStyleSheet("background-color: rgb(220,220,220); border: 1px solid black;")
        self.label.setGraphicsEffect(self.shadow)

        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton_1.setText(_translate("Dialog", "Back"))

    def back_event(self):
        """
        on clicking the linked button, it opens the corresponding page and call for clearing the selections
        """
        self.goto('main')
