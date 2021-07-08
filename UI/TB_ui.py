# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './UI/TB_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.web_widget = QtWidgets.QWidget(self.centralwidget)
        self.web_widget.setObjectName("web_widget")
        self.web_widget_Layout = QtWidgets.QVBoxLayout(self.web_widget)
        self.web_widget_Layout.setObjectName("web_widget_Layout")
        self.verticalLayout.addWidget(self.web_widget)
        self.refresh_btn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(26)
        self.refresh_btn.setFont(font)
        self.refresh_btn.setObjectName("refresh_btn")
        self.verticalLayout.addWidget(self.refresh_btn)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "TensorBoard"))
        self.refresh_btn.setText(_translate("MainWindow", "手动刷新"))

