import numpy as np

class OthelloBoard:
    def __init__(self, size=8):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        # 초기 보드 설정 (중앙에 흰색, 검은색 돌 배치)
        mid = size // 2
        self.board[mid-1:mid+1, mid-1:mid+1] = [[1, -1], [-1, 1]]
        self.current_player = 1  # -1: 검은색, 1: 흰색
        self.directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

    def is_on_board(self, row, col):
        """보드 내부인지 확인"""
        return 0 <= row < self.size and 0 <= col < self.size

    def is_valid_move(self, row, col):
        """해당 위치에 돌을 놓을 수 있는지 확인"""
        # 이미 돌이 있는 칸은 놓을 수 없음
        if self.board[row, col] != 0:
            return False

        # 8방향 탐색
        for dx, dy in self.directions:
            x, y = row + dx, col + dy
            # 첫 번째 인접 칸이 상대방 돌인지 확인
            if self.is_on_board(x, y) and self.board[x, y] == -self.current_player:
                # 계속해서 상대방 돌이 있는지 확인
                while self.is_on_board(x, y) and self.board[x, y] == -self.current_player:
                    x += dx
                    y += dy
                # 마지막에 현재 플레이어의 돌이 있으면 유효한 이동
                if self.is_on_board(x, y) and self.board[x, y] == self.current_player:
                    return True
        return False

    def make_move(self, row, col):
        """돌을 놓고 뒤집기"""
        if not self.is_valid_move(row, col):
            return False

        # 돌 놓기
        self.board[row, col] = self.current_player

        # 8방향 탐색하며 뒤집기
        for dx, dy in self.directions:
            x, y = row + dx, col + dy
            to_flip = []

            # 상대방 돌 찾기
            while self.is_on_board(x, y) and self.board[x, y] == -self.current_player:
                to_flip.append((x, y))
                x += dx
                y += dy

            # 마지막 돌이 현재 플레이어 돌이면 중간 돌들 뒤집기
            if self.is_on_board(x, y) and self.board[x, y] == self.current_player:
                for fx, fy in to_flip:
                    self.board[fx, fy] = self.current_player

        return True

    def get_valid_moves(self):
        """현재 플레이어의 가능한 모든 이동 반환"""
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.is_valid_move(row, col):
                    moves.append((row, col))
        return moves

    def count_stones(self):
        """현재 보드의 돌 개수 세기"""
        white_stones = np.sum(self.board == 1)
        black_stones = np.sum(self.board == -1)
        return white_stones, black_stones

    def is_game_over(self):
        """게임 종료 조건 확인"""
        # 보드가 꽉 찼는지 확인
        if np.count_nonzero(self.board == 0) == 0:
            return True

        # 양쪽 모두 둘 수 있는 수가 없는지 확인
        current_player_backup = self.current_player
        current_moves = self.get_valid_moves()
        
        # 플레이어 교체
        self.current_player *= -1
        opponent_moves = self.get_valid_moves()
        
        # 원래 플레이어로 복귀
        self.current_player = current_player_backup

        return len(current_moves) == 0 and len(opponent_moves) == 0

    def get_winner(self):
        """승자 결정"""
        white_stones, black_stones = self.count_stones()
        if white_stones > black_stones:
            return 1  # 흰색 승리
        elif black_stones > white_stones:
            return -1  # 검은색 승리
        else:
            return 0  # 무승부

