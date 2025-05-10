import numpy as np
import random

class BaseOthelloAgent:
    def __init__(self, board_size=8):
        """
        기본 오델로 강화학습 에이전트 초기화
        
        :param board_size: 보드 크기 (기본값 8x8)
        """
        self.board_size = board_size
        self.q_table = {}  # Q-learning을 위한 Q-테이블
    
    def get_state_key(self, board_state):
        """
        보드 상태를 문자열 키로 변환
        
        :param board_state: 현재 보드 상태 (numpy 배열)
        :return: 보드 상태의 문자열 표현
        """
        return str(board_state.flatten())
    
    def choose_action(self, board, valid_moves, learning_mode=True):
        """
        행동 선택 (탐험 vs 활용)
        
        :param board: 현재 보드 상태
        :param valid_moves: 유효한 이동 위치들
        :param learning_mode: 학습 모드인지 여부
        :return: 선택된 이동 위치
        """
        if not valid_moves:
            return None
        
        # 탐험 vs 활용 전략
        if learning_mode and random.random() < 0.1:  # 10% 탐험 확률
            return random.choice(valid_moves)
        
        # Q-테이블을 기반으로 최적의 행동 선택
        state_key = self.get_state_key(board)
        best_move = None
        max_q_value = float('-inf')
        
        for move in valid_moves:
            move_key = f"{state_key}_{move[0]}_{move[1]}"
            q_value = self.q_table.get(move_key, 0)
            
            if q_value > max_q_value:
                max_q_value = q_value
                best_move = move
        
        return best_move or random.choice(valid_moves)
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Q-값 업데이트
        
        :param state: 이전 상태
        :param action: 취한 행동
        :param reward: 보상
        :param next_state: 다음 상태
        """
        learning_rate = 0.1
        discount_factor = 0.9
        
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        action_key = f"{state_key}_{action[0]}_{action[1]}"
        
        # 현재 Q-값
        current_q = self.q_table.get(action_key, 0)
        
        # 다음 상태의 최대 Q-값 계산
        max_next_q = max(
            self.q_table.get(f"{next_state_key}_{move[0]}_{move[1]}", 0) 
            for move in [(r, c) for r in range(self.board_size) for c in range(self.board_size)]
        )
        
        # Q-값 업데이트
        new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
        self.q_table[action_key] = new_q

    def get_reward(self, board, player):
        """
        현재 보드 상태에 대한 보상 계산
        
        :param board: 현재 보드 상태
        :param player: 현재 플레이어 (-1: 흑, 1: 백)
        :return: 보상 값
        """
        white_stones, black_stones = np.count_nonzero(board == 1), np.count_nonzero(board == -1)
        
        if player == -1:  # 흑 플레이어 기준
            return black_stones - white_stones
        else:  # 백 플레이어 기준
            return white_stones - black_stones
