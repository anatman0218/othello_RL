import numpy as np

class OthelloBoard:
    def __init__(self, size=8):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        # 초기 보드 설정 (중앙에 흰색, 검은색 돌 배치)
        mid = size // 2
        self.board[mid-1:mid+1, mid-1:mid+1] = [[1, -1], [-1, 1]]
        self.current_player = 1  # 1: 흰색, -1: 검은색

    def is_valid_move(self, row, col):
        # 이동 가능한 위치인지 확인하는 로직
        pass

    def make_move(self, row, col):
        # 돌을 놓고 뒤집는 로직
        pass

    def get_valid_moves(self):
        # 현재 플레이어의 가능한 모든 이동 반환
        pass

    def is_game_over(self):
        # 게임 종료 조건 확인
        pass
