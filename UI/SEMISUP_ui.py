# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './UI/SEMISUP_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1022, 806)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.intime_canvas = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.intime_canvas.sizePolicy().hasHeightForWidth())
        self.intime_canvas.setSizePolicy(sizePolicy)
        self.intime_canvas.setMinimumSize(QtCore.QSize(980, 380))
        self.intime_canvas.setObjectName("intime_canvas")
        self.intime_layout = QtWidgets.QVBoxLayout(self.intime_canvas)
        self.intime_layout.setObjectName("intime_layout")
        self.verticalLayout.addWidget(self.intime_canvas)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.tabWidget.setFont(font)
        self.tabWidget.setUsesScrollButtons(False)
        self.tabWidget.setTabBarAutoHide(False)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout = QtWidgets.QGridLayout(self.tab)
        self.gridLayout.setObjectName("gridLayout")
        self.epochs = QtWidgets.QLineEdit(self.tab)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.epochs.setFont(font)
        self.epochs.setAlignment(QtCore.Qt.AlignCenter)
        self.epochs.setPlaceholderText("")
        self.epochs.setObjectName("epochs")
        self.gridLayout.addWidget(self.epochs, 0, 1, 1, 3)
        self.label_6 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 2, 0, 1, 1)
        self.batch_size = QtWidgets.QLineEdit(self.tab)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.batch_size.setFont(font)
        self.batch_size.setAlignment(QtCore.Qt.AlignCenter)
        self.batch_size.setPlaceholderText("")
        self.batch_size.setObjectName("batch_size")
        self.gridLayout.addWidget(self.batch_size, 2, 1, 1, 3)
        self.label_3 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 1)
        self.latent_dim = QtWidgets.QLineEdit(self.tab)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.latent_dim.setFont(font)
        self.latent_dim.setAlignment(QtCore.Qt.AlignCenter)
        self.latent_dim.setObjectName("latent_dim")
        self.gridLayout.addWidget(self.latent_dim, 3, 1, 1, 3)
        self.singal2pic = QtWidgets.QCheckBox(self.tab)
        self.singal2pic.setObjectName("singal2pic")
        self.gridLayout.addWidget(self.singal2pic, 4, 0, 1, 1)
        self.save_last_only = QtWidgets.QCheckBox(self.tab)
        self.save_last_only.setObjectName("save_last_only")
        self.gridLayout.addWidget(self.save_last_only, 4, 1, 1, 2)
        self.trai_button = QtWidgets.QPushButton(self.tab)
        font = QtGui.QFont()
        font.setPointSize(28)
        self.trai_button.setFont(font)
        self.trai_button.setObjectName("trai_button")
        self.gridLayout.addWidget(self.trai_button, 4, 3, 1, 1)
        self.load_labeled = QtWidgets.QPushButton(self.tab)
        font = QtGui.QFont()
        font.setPointSize(28)
        self.load_labeled.setFont(font)
        self.load_labeled.setObjectName("load_labeled")
        self.gridLayout.addWidget(self.load_labeled, 5, 0, 1, 1)
        self.load_unlabeled = QtWidgets.QPushButton(self.tab)
        font = QtGui.QFont()
        font.setPointSize(28)
        self.load_unlabeled.setFont(font)
        self.load_unlabeled.setObjectName("load_unlabeled")
        self.gridLayout.addWidget(self.load_unlabeled, 5, 1, 1, 1)
        self.convert2pic = QtWidgets.QPushButton(self.tab)
        self.convert2pic.setEnabled(False)
        self.convert2pic.setObjectName("convert2pic")
        self.gridLayout.addWidget(self.convert2pic, 5, 2, 1, 1)
        self.show_tb_btn = QtWidgets.QPushButton(self.tab)
        self.show_tb_btn.setEnabled(False)
        self.show_tb_btn.setObjectName("show_tb_btn")
        self.gridLayout.addWidget(self.show_tb_btn, 5, 3, 1, 1)
        self.label = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.train_progressBar = QtWidgets.QProgressBar(self.tab)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.train_progressBar.setFont(font)
        self.train_progressBar.setStatusTip("")
        self.train_progressBar.setProperty("value", 20)
        self.train_progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.train_progressBar.setObjectName("train_progressBar")
        self.gridLayout.addWidget(self.train_progressBar, 6, 0, 1, 4)
        self.tabWidget.addTab(self.tab, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_4)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.predict_b = QtWidgets.QPushButton(self.tab_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predict_b.sizePolicy().hasHeightForWidth())
        self.predict_b.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.predict_b.setFont(font)
        self.predict_b.setObjectName("predict_b")
        self.gridLayout_2.addWidget(self.predict_b, 9, 3, 1, 1)
        self.model_b = QtWidgets.QPushButton(self.tab_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.model_b.sizePolicy().hasHeightForWidth())
        self.model_b.setSizePolicy(sizePolicy)
        self.model_b.setObjectName("model_b")
        self.gridLayout_2.addWidget(self.model_b, 9, 1, 1, 1)
        self.load_test_b = QtWidgets.QPushButton(self.tab_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.load_test_b.sizePolicy().hasHeightForWidth())
        self.load_test_b.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.load_test_b.setFont(font)
        self.load_test_b.setObjectName("load_test_b")
        self.gridLayout_2.addWidget(self.load_test_b, 9, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.tab_4)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.label_5.setFont(font)
        self.label_5.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 4, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.tab_4)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 5, 0, 1, 1)
        self.class_txt = QtWidgets.QLineEdit(self.tab_4)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.class_txt.setFont(font)
        self.class_txt.setAlignment(QtCore.Qt.AlignCenter)
        self.class_txt.setReadOnly(True)
        self.class_txt.setObjectName("class_txt")
        self.gridLayout_2.addWidget(self.class_txt, 4, 1, 1, 3)
        self.class_pro = QtWidgets.QLineEdit(self.tab_4)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.class_pro.setFont(font)
        self.class_pro.setAlignment(QtCore.Qt.AlignCenter)
        self.class_pro.setReadOnly(True)
        self.class_pro.setObjectName("class_pro")
        self.gridLayout_2.addWidget(self.class_pro, 5, 1, 1, 3)
        self.tabWidget.addTab(self.tab_4, "")
        self.verticalLayout.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.tabWidget, self.epochs)
        MainWindow.setTabOrder(self.epochs, self.batch_size)
        MainWindow.setTabOrder(self.batch_size, self.latent_dim)
        MainWindow.setTabOrder(self.latent_dim, self.trai_button)
        MainWindow.setTabOrder(self.trai_button, self.show_tb_btn)
        MainWindow.setTabOrder(self.show_tb_btn, self.class_txt)
        MainWindow.setTabOrder(self.class_txt, self.class_pro)
        MainWindow.setTabOrder(self.class_pro, self.load_test_b)
        MainWindow.setTabOrder(self.load_test_b, self.model_b)
        MainWindow.setTabOrder(self.model_b, self.predict_b)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "半监督学习"))
        self.label_6.setStatusTip(_translate("MainWindow", "Batch Size 大小"))
        self.label_6.setText(_translate("MainWindow", "batch_size:"))
        self.label_3.setStatusTip(_translate("MainWindow", "隐层结点个数， 建议500-1000"))
        self.label_3.setText(_translate("MainWindow", "latent_dim:"))
        self.latent_dim.setPlaceholderText(_translate("MainWindow", "1000"))
        self.singal2pic.setStatusTip(_translate("MainWindow", "是否将一维信号转为时频图片信号，勾选才可以转时频图"))
        self.singal2pic.setText(_translate("MainWindow", "signal2pic"))
        self.save_last_only.setStatusTip(_translate("MainWindow", "是否仅保存最后一个epoch的模型"))
        self.save_last_only.setText(_translate("MainWindow", "Save last only"))
        self.trai_button.setStatusTip(_translate("MainWindow", "开始训练"))
        self.trai_button.setText(_translate("MainWindow", "训练"))
        self.load_labeled.setStatusTip(_translate("MainWindow", "加载前处理所存的有标签.npy"))
        self.load_labeled.setText(_translate("MainWindow", "加载有标签样本"))
        self.load_unlabeled.setStatusTip(_translate("MainWindow", "加载前处理所存的无标签.npy"))
        self.load_unlabeled.setText(_translate("MainWindow", "加载无标签样本"))
        self.convert2pic.setStatusTip(_translate("MainWindow", "singal转为时频图"))
        self.convert2pic.setText(_translate("MainWindow", "转时频图"))
        self.show_tb_btn.setStatusTip(_translate("MainWindow", "展开TensorBoard具体训练过程"))
        self.show_tb_btn.setText(_translate("MainWindow", "展开训练过程"))
        self.label.setStatusTip(_translate("MainWindow", "训练迭代轮数"))
        self.label.setText(_translate("MainWindow", "epochs:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "训练"))
        self.predict_b.setStatusTip(_translate("MainWindow", "信号转时频图， 然后开始检测"))
        self.predict_b.setText(_translate("MainWindow", "检测"))
        self.model_b.setStatusTip(_translate("MainWindow", "加载训练好的.h5模型"))
        self.model_b.setText(_translate("MainWindow", "加载模型"))
        self.load_test_b.setStatusTip(_translate("MainWindow", "加载待测试数据"))
        self.load_test_b.setText(_translate("MainWindow", "加载数据"))
        self.label_5.setText(_translate("MainWindow", "类别:"))
        self.label_4.setText(_translate("MainWindow", "概率："))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "检测"))
