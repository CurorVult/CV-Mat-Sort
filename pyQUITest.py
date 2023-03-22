import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
from PyQt5.QtGui import QIcon

class SimpleWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Simple PyQt5 Window')
        self.setWindowIcon(QIcon('icon.png'))
        self.setGeometry(100, 100, 1800, 1000)

        button = QPushButton('Capture', self)
        button.setToolTip('Click to show a message box')
        button.clicked.connect(self.show_message)
        button.resize(button.sizeHint())
        button.move(150, 350)

    def show_message(self):
        QMessageBox.information(self, 'Message', 'Hello, PyQt5!')

def main():
    app = QApplication(sys.argv)
    window = SimpleWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()