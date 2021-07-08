from PyQt5.Qt import *
from UI.MAIN_ui import Ui_MainWindow
import sys
from PRE_main import PRE_wind
from SUP_main import SUP_Wind
from SEMISUP_main import SEMISUP_Wind
from UNSUP_main import UNSUP_Wind
from SUFF_main import SUFF_Wind


class Main_MainWind(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        # self.setFixedSize(520, 520)

        self.pre_btn.clicked.connect(self.call_pre)
        self.sup_btn.clicked.connect(self.call_sup)
        self.unsup_btn.clicked.connect(self.call_unsup)
        self.semisup_btn.clicked.connect(self.call_semi_sup)
        self.suffix_btn.clicked.connect(self.call_suffix)

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
    app = QApplication(sys.argv)
    wind = Main_MainWind()
    wind.show()
    sys.exit(app.exec_())
