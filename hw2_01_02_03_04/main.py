from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import MainWindow as ui
import os

from Q1.Q1 import Question1
from Q2.Q2 import Question2
from Q3.Q3 import Question3
from Q4.Q4 import Question4


class Main(QtWidgets.QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.imagePath = None
        self.videoPath = None
        self.dirPath = None

        # load data
        self.pushButtonLoadVideo.clicked.connect(self.getVideoPath)
        self.pushButtonLoadImage.clicked.connect(self.getImagePath)
        self.pushButtonLoadFolder.clicked.connect(self.selectDir)

        # question 1
        self.pushButtonBackgroundSubtraction.clicked.connect(lambda: Q1Object.backgroundSubtraction(self.videoPath))

        # question 2
        self.pushButtonPreprocessing.clicked.connect(lambda: Q2Object.preprocessing(self.videoPath))
        self.pushButtonVideoTracking.clicked.connect(lambda: Q2Object.videoTracking(self.videoPath))

        # question 3
        self.pushButtonPerspectiveTransform.clicked.connect(lambda: Q3Object.perspectiveTransform(self.videoPath, self.imagePath))

        # question 4
        self.pushButtonImageReconstruction.clicked.connect(lambda: Q4Object.imageReconstruction(self.dirPath))
        self.pushButtonComputeTheReconstructionError.clicked.connect(Q4Object.showReconstructionError)

    
    def selectVideo(self):
        fileName = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(None, caption='Choose a File', directory='C:\\', filter='Video Files (*.mp4)')[0])  # get turple[0] which is file name
        return fileName
    
    def getVideoPath(self):
        self.videoPath = self.selectVideo()

    def selectImage(self):
        fileName = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(None, caption='Choose a File', directory='C:\\', filter='Image Files (*.png *.jpg *.bmp)')[0])  # get turple[0] which is file name
        return fileName
    
    def getImagePath(self):
        self.imagePath = self.selectImage()

    def selectDir(self):
        self.dirPath = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getExistingDirectory(None, caption='Select a folder:', directory='C:\\', options=QtWidgets.QFileDialog.ShowDirsOnly))

    # overide to force exit
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        os._exit(0)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    Q1Object = Question1()
    Q2Object = Question2()
    Q3Object = Question3()
    Q4Object = Question4()
    window = Main()
    window.show()
    sys.exit(app.exec_())