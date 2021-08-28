from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PRE_main import PRE_wind
from SUP_main import SUP_Wind
from SEMISUP_main import SEMISUP_Wind
from UNSUP_main import UNSUP_Wind
from SUFF_main import SUFF_Wind


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1101, 716)
        MainWindow.setStyleSheet("#MainWindow{border-image:url(UI/ico/background1.jpg)}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.pre_btn = QtWidgets.QAction(QtGui.QIcon("UI/ico/北冥.ico"), "前处理")
        self.sup_btn = QtWidgets.QAction(QtGui.QIcon("UI/ico/天尊.ico"), "监督学习")
        self.semisup_btn = QtWidgets.QAction(QtGui.QIcon("UI/ico/炽凰.ico"), "半监督学习")
        self.unsup_btn = QtWidgets.QAction(QtGui.QIcon("UI/ico/苍魂.ico"), "无监督学习")
        self.suffix_btn = QtWidgets.QAction(QtGui.QIcon("UI/ico/魔君.ico"), "后处理")
        self.toolBar.addActions([self.pre_btn, self.sup_btn, self.unsup_btn, self.semisup_btn, self.suffix_btn])
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "半监督、无监督深度学习诊断软件"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.pre_btn.triggered.connect(self.call_pre)
        self.sup_btn.triggered.connect(self.call_sup)
        self.unsup_btn.triggered.connect(self.call_unsup)
        self.semisup_btn.triggered.connect(self.call_semi_sup)
        self.suffix_btn.triggered.connect(self.call_suffix)

    def call_pre(self):
        # self.hide()
        self.win_pre = PRE_wind()
        self.win_pre.show()

    def call_sup(self):
        # self.hide()
        self.win_sup = SUP_Wind()
        self.win_sup.show()

    def call_unsup(self):
        # self.hide()
        self.win_unsup = UNSUP_Wind()
        self.win_unsup.show()

    def call_semi_sup(self):
        # self.hide()
        self.win_semi_sup = SEMISUP_Wind()
        self.win_semi_sup.show()

    def call_suffix(self):
        self.win_suffix = SUFF_Wind()
        self.win_suffix.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
