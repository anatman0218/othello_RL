import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QPushButton, QWidget, QLabel
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
from board import OthelloBoard

class OthelloUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.board = OthelloBoard()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Othello Game')
        self.setGeometry(100, 100, 600, 600)

        # 중앙 위젯과 그리드 레이아웃 생성
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        grid_layout = QGridLayout()
        central_widget.setLayout(grid_layout)

        # 보드 버튼 생성
        self.board_buttons = []
        for row in range(self.board.size):
            row_buttons = []
            for col in range(self.board.size):
                button = QPushButton()
                button.setFixedSize(60, 60)
                button.clicked.connect(lambda checked, r=row, c=col: self.on_cell_clicked(r, c))
                grid_layout.addWidget(button, row, col)
                row_buttons.append(button)
            self.board_buttons.append(row_buttons)

        self.update_board_ui()

    def update_board_ui(self):
        # 보드 상태에 따라 버튼 색상 업데이트
        for row in range(self.board.size):
            for col in range(self.board.size):
                cell_value = self.board.board[row, col]
                button = self.board_buttons[row][col]
                if cell_value == 1:
                    button.setStyleSheet("background-color: white; border: 1px solid black;")
                elif cell_value == -1:
                    button.setStyleSheet("background-color: black; border: 1px solid black;")
                else:
                    button.setStyleSheet("background-color: green; border: 1px solid black;")

    def on_cell_clicked(self, row, col):
        # 셀 클릭 시 게임 로직 처리
        if self.board.is_valid_move(row, col):
            self.board.make_move(row, col)
            self.update_board_ui()

def main():
    app = QApplication(sys.argv)
    game = OthelloUI()
    game.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
