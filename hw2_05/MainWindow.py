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
        MainWindow.resize(2026, 1202)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupResnet50 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupResnet50.setGeometry(QtCore.QRect(50, 60, 341, 1111))
        self.groupResnet50.setObjectName("groupResnet50")
        self.pushButtonShowImages = QtWidgets.QPushButton(self.groupResnet50)
        self.pushButtonShowImages.setGeometry(QtCore.QRect(30, 230, 281, 51))
        self.pushButtonShowImages.setObjectName("pushButtonShowImages")
        self.pushButtonShowModelStructure = QtWidgets.QPushButton(self.groupResnet50)
        self.pushButtonShowModelStructure.setGeometry(QtCore.QRect(30, 590, 281, 51))
        self.pushButtonShowModelStructure.setObjectName("pushButtonShowModelStructure")
        self.pushButtonShowDistribution = QtWidgets.QPushButton(self.groupResnet50)
        self.pushButtonShowDistribution.setGeometry(QtCore.QRect(30, 410, 281, 51))
        self.pushButtonShowDistribution.setObjectName("pushButtonShowDistribution")
        self.pushButtonShowComparsion = QtWidgets.QPushButton(self.groupResnet50)
        self.pushButtonShowComparsion.setGeometry(QtCore.QRect(30, 770, 281, 51))
        self.pushButtonShowComparsion.setObjectName("pushButtonShowComparsion")
        self.pushButtonShowInference = QtWidgets.QPushButton(self.groupResnet50)
        self.pushButtonShowInference.setGeometry(QtCore.QRect(30, 950, 281, 51))
        self.pushButtonShowInference.setObjectName("pushButtonShowInference")
        self.pushButtonLoadImage = QtWidgets.QPushButton(self.groupResnet50)
        self.pushButtonLoadImage.setGeometry(QtCore.QRect(30, 60, 281, 51))
        self.pushButtonLoadImage.setObjectName("pushButtonLoadImage")
        self.photo = QtWidgets.QLabel(self.centralwidget)
        self.photo.setGeometry(QtCore.QRect(530, 200, 1301, 981))
        self.photo.setText("")
        # self.photo.setPixmap(QtGui.QPixmap("../cvdl_hw1_04/Dataset_CvDl_Hw1_2/Q4_images/traffics.png"))
        self.photo.setAlignment(QtCore.Qt.AlignCenter)
        self.photo.setObjectName("photo")
        self.textArea = QtWidgets.QLabel(self.centralwidget)
        self.textArea.setGeometry(QtCore.QRect(530, 920, 1301, 111))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(16)
        self.textArea.setFont(font)
        self.textArea.setText("")
        self.textArea.setTextFormat(QtCore.Qt.AutoText)
        self.textArea.setAlignment(QtCore.Qt.AlignCenter)
        self.textArea.setObjectName("textArea")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "2022 CvDl Hw2_05"))
        self.groupResnet50.setTitle(_translate("MainWindow", "5. Resnet50"))
        self.pushButtonShowImages.setText(_translate("MainWindow", "1. Show Images"))
        self.pushButtonShowModelStructure.setText(_translate("MainWindow", "3. Show Model Structure"))
        self.pushButtonShowDistribution.setText(_translate("MainWindow", "2. Show Distribution"))
        self.pushButtonShowComparsion.setText(_translate("MainWindow", "4. Show Comparsion"))
        self.pushButtonShowInference.setText(_translate("MainWindow", "5. Inference"))
        self.pushButtonLoadImage.setText(_translate("MainWindow", "Load Image"))
