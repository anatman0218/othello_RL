import sys
import os
import locale

from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QPushButton, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QColor, QFont, QFontDatabase
from PyQt5.QtCore import Qt

# 한글 인코딩 및 폰트 설정
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.board import OthelloBoard

class OthelloUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.board = OthelloBoard()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('오델로 게임')
        self.setGeometry(100, 100, 700, 700)
        
        # WSL 한글 렌더링 설정
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

        # 중앙 위젯과 메인 레이아웃 생성
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # 폰트 설정
        korean_font = QFont('D2Coding', 14)
        korean_font.setHintingPreference(QFont.PreferNoHinting)

        # 현재 플레이어 표시 라벨
        self.player_label = QLabel()
        self.player_label.setFont(korean_font)
        self.player_label.setStyleSheet('font-family: D2Coding;')
        main_layout.addWidget(self.player_label)

        # 점수 표시 라벨
        self.score_label = QLabel()
        self.score_label.setFont(korean_font)
        self.score_label.setStyleSheet('font-family: D2Coding;')
        main_layout.addWidget(self.score_label)

        # 보드 그리드 레이아웃
        grid_layout = QGridLayout()
        main_layout.addLayout(grid_layout)

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
        valid_moves = self.board.get_valid_moves()
        white_stones, black_stones = self.board.count_stones()

        # Current player and score display
        player_text = 'Black Turn' if self.board.current_player == -1 else 'White Turn'
        self.player_label.setText('Current Player: {}'.format(player_text))
        self.score_label.setText('Black: {} | White: {}'.format(black_stones, white_stones))
        
        # Debug information
        print(f'System Encoding: {sys.getdefaultencoding()}')
        print(f'Locale: {locale.getlocale()}')
        
        # Label font settings
        self.player_label.setStyleSheet('font-family: Arial; font-size: 14px;')
        self.score_label.setStyleSheet('font-family: Arial; font-size: 14px;')

        for row in range(self.board.size):
            for col in range(self.board.size):
                cell_value = self.board.board[row, col]
                button = self.board_buttons[row][col]

                # 기본 보드 색상
                button.setStyleSheet("background-color: darkgreen; border: 1px solid black;")
                button.setText('')  # Clear any previous text
                
                if cell_value == 1:  # White stone
                    button.setStyleSheet("""
                        background-color: darkgreen; 
                        border: 2px solid white; 
                        border-radius: 30px; 
                        background-image: radial-gradient(circle at 30% 30%, white, lightgray); 
                        color: transparent;
                        box-shadow: 0 0 5px rgba(255, 255, 255, 0.5);
                    """)
                elif cell_value == -1:  # Black stone
                    button.setStyleSheet("""
                        background-color: darkgreen; 
                        border: 2px solid black; 
                        border-radius: 30px; 
                        background-image: radial-gradient(circle at 30% 30%, black, darkgray); 
                        color: transparent;
                        box-shadow: 0 0 5px rgba(0, 0, 0, 0.5);
                    """)
                elif (row, col) in valid_moves:  # Valid moves
                    button.setStyleSheet("""
                        background-color: rgba(0, 255, 0, 50); 
                        border: 1px solid green;
                        border-radius: 10px;
                    """)

        # 게임 종료 확인
        if self.board.is_game_over():
            winner = self.board.get_winner()
            if winner == 1:
                self.player_label.setText("흰색 승리!")
            elif winner == -1:
                self.player_label.setText("검은색 승리!")
            else:
                self.player_label.setText("무승부!")

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
