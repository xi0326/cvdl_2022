# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(417, 1202)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBackgroundSubtraction = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBackgroundSubtraction.setGeometry(QtCore.QRect(30, 240, 341, 131))
        self.groupBackgroundSubtraction.setObjectName("groupBackgroundSubtraction")
        self.pushButtonBackgroundSubtraction = QtWidgets.QPushButton(self.groupBackgroundSubtraction)
        self.pushButtonBackgroundSubtraction.setGeometry(QtCore.QRect(30, 40, 281, 51))
        self.pushButtonBackgroundSubtraction.setObjectName("pushButtonBackgroundSubtraction")
        self.pushButtonLoadVideo = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonLoadVideo.setGeometry(QtCore.QRect(60, 20, 281, 51))
        self.pushButtonLoadVideo.setObjectName("pushButtonLoadVideo")
        self.pushButtonLoadImage = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonLoadImage.setGeometry(QtCore.QRect(60, 80, 281, 51))
        self.pushButtonLoadImage.setObjectName("pushButtonLoadImage")
        self.pushButtonLoadFolder = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonLoadFolder.setGeometry(QtCore.QRect(60, 140, 281, 51))
        self.pushButtonLoadFolder.setObjectName("pushButtonLoadFolder")
        self.groupOpticalFlow = QtWidgets.QGroupBox(self.centralwidget)
        self.groupOpticalFlow.setGeometry(QtCore.QRect(30, 430, 341, 221))
        self.groupOpticalFlow.setObjectName("groupOpticalFlow")
        self.pushButtonPreprocessing = QtWidgets.QPushButton(self.groupOpticalFlow)
        self.pushButtonPreprocessing.setGeometry(QtCore.QRect(30, 40, 281, 51))
        self.pushButtonPreprocessing.setObjectName("pushButtonPreprocessing")
        self.pushButtonVideoTracking = QtWidgets.QPushButton(self.groupOpticalFlow)
        self.pushButtonVideoTracking.setGeometry(QtCore.QRect(30, 130, 281, 51))
        self.pushButtonVideoTracking.setObjectName("pushButtonVideoTracking")
        self.groupPerspectiveTransform = QtWidgets.QGroupBox(self.centralwidget)
        self.groupPerspectiveTransform.setGeometry(QtCore.QRect(30, 700, 341, 131))
        self.groupPerspectiveTransform.setObjectName("groupPerspectiveTransform")
        self.pushButtonPerspectiveTransform = QtWidgets.QPushButton(self.groupPerspectiveTransform)
        self.pushButtonPerspectiveTransform.setGeometry(QtCore.QRect(30, 40, 281, 51))
        self.pushButtonPerspectiveTransform.setObjectName("pushButtonPerspectiveTransform")
        self.groupPCA = QtWidgets.QGroupBox(self.centralwidget)
        self.groupPCA.setGeometry(QtCore.QRect(30, 910, 341, 221))
        self.groupPCA.setObjectName("groupPCA")
        self.pushButtonImageReconstruction = QtWidgets.QPushButton(self.groupPCA)
        self.pushButtonImageReconstruction.setGeometry(QtCore.QRect(30, 40, 281, 51))
        self.pushButtonImageReconstruction.setObjectName("pushButtonImageReconstruction")
        self.pushButtonComputeTheReconstructionError = QtWidgets.QPushButton(self.groupPCA)
        self.pushButtonComputeTheReconstructionError.setGeometry(QtCore.QRect(30, 130, 281, 51))
        self.pushButtonComputeTheReconstructionError.setObjectName("pushButtonComputeTheReconstructionError")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "2022 CvDl Hw2"))
        self.groupBackgroundSubtraction.setTitle(_translate("MainWindow", "1. Background Subtraction"))
        self.pushButtonBackgroundSubtraction.setText(_translate("MainWindow", "1. Background Subtraction"))
        self.pushButtonLoadVideo.setText(_translate("MainWindow", "Load Video"))
        self.pushButtonLoadImage.setText(_translate("MainWindow", "Load Image"))
        self.pushButtonLoadFolder.setText(_translate("MainWindow", "Load Folder"))
        self.groupOpticalFlow.setTitle(_translate("MainWindow", "2. Optical Flow"))
        self.pushButtonPreprocessing.setText(_translate("MainWindow", "2.1 Preprocessing"))
        self.pushButtonVideoTracking.setText(_translate("MainWindow", "2.2 Video Tracking"))
        self.groupPerspectiveTransform.setTitle(_translate("MainWindow", "3. Perspective Transform"))
        self.pushButtonPerspectiveTransform.setText(_translate("MainWindow", "3.1 Perspective Transform"))
        self.groupPCA.setTitle(_translate("MainWindow", "4.1 PCA"))
        self.pushButtonImageReconstruction.setText(_translate("MainWindow", "4.1 Image Reconstruction"))
        self.pushButtonComputeTheReconstructionError.setText(_translate("MainWindow", "4.2 Compute the reconstruction error"))
