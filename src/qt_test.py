import sys
import os

from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtCore import Qt

def main():
    # 디버깅용 환경 변수 출력
    print(f"DISPLAY: {os.environ.get('DISPLAY', 'Not set')}")
    print(f"QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM', 'Not set')}")
    
    # Qt 플랫폼 플러그인 디버깅
    print("Available Qt platform plugins:", QApplication.platformName())

    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle('WSL PyQt Test')
    
    # 창 위치와 크기 조정
    window.setGeometry(100, 100, 250, 150)
    label = QLabel('Hello, WSL PyQt!', parent=window)
    label.setAlignment(Qt.AlignCenter)
    
    window.show()
    return app.exec_()

if __name__ == '__main__':
    main()
