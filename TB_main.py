from PyQt5.Qt import *
from UI.TB_ui import Ui_MainWindow
from PyQt5.QtWebEngineWidgets import *
import sys


class TB_MainWind(Ui_MainWindow, QMainWindow):
    def __init__(self, web_site):
        super(TB_MainWind, self).__init__()
        self.setupUi(self)

        self.web_site = web_site

        self.tb_web = QWebEngineView(self)
        self.web_widget_Layout.addWidget(self.tb_web)

        self.tb_web.load(QUrl(self.web_site))
        self.refresh_btn.clicked.connect(self.refresh)

    def refresh(self):
        self.tb_web.load(QUrl(self.web_site))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    wind = TB_MainWind('http://www.baidu.com')
    wind.show()
    sys.exit(app.exec_())
