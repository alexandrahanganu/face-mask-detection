from PyQt5 import QtWidgets, QtCore, QtGui


class PageWindow(QtWidgets.QMainWindow):
    gotoSignal = QtCore.pyqtSignal(str)

    def goto(self, name):
        self.gotoSignal.emit(name)

    @staticmethod
    def add_fonts(instance):
        """
        adds the fonts to the QFontDatabase and also sets the stylesheet for the instance
        """
        QtGui.QFontDatabase.addApplicationFont("fonts/Comfortaa/Comfortaa-SemiBold.ttf")
        QtGui.QFontDatabase.addApplicationFont("fonts/Montserrat/Montserrat-LightItalic.ttf")
        stylesheet = open('stylesheets/styles.qss').read()
        instance.setStyleSheet(stylesheet)
