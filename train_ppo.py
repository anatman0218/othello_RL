import os
import logging
from datetime import datetime

import numpy as np
from src.board import OthelloBoard
from rl_agent.ppo_agent import PPOAgent

# 로깅 설정
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f'train_ppo_{current_time}.log')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def record_game(white_agent, black_agent, log_file):
    """
    마지막 경기 기록
    """
    with open(log_file, 'w') as f:
        board = OthelloBoard()
        f.write(f"Initial Board:\n{board}\n")
        f.write(f"Initial valid moves: {board.get_valid_moves()}\n")
        f.write(f"Initial board state:\n{board.board}\n\n")
        
        current_agent = white_agent
        opponent_agent = black_agent
        turn = 0
        
        while not board.is_game_over():
            turn += 1
            f.write(f"Turn {turn}:\n")
            valid_moves = board.get_valid_moves()
            f.write(f"Current player: {'White' if board.current_player == 1 else 'Black'}\n")
            f.write(f"Valid moves: {valid_moves}\n")
            
            if not valid_moves:
                f.write("No valid moves. Skipping turn.\n")
                current_agent, opponent_agent = opponent_agent, current_agent
                board.current_player *= -1
                continue
            
            action = current_agent.choose_action(board.board, valid_moves, learning_mode=False)
            f.write(f"Chosen action: {action}\n")
            
            board.make_move(action[0], action[1])
            
            f.write(f"Board after move:\n{board}\n")
            f.write(f"Board state:\n{board.board}\n\n")
            
            current_agent, opponent_agent = opponent_agent, current_agent
            board.current_player *= -1
        
        winner = board.get_winner()
        if winner == 1:
            f.write("White wins!\n")
        elif winner == -1:
            f.write("Black wins!\n")
        else:
            f.write("Draw!\n")
        
        white_stones, black_stones = board.count_stones()
        f.write(f"Final stone count - White: {white_stones}, Black: {black_stones}")

def train_ppo_agents(num_episodes=10):
    """
    두 PPO 에이전트 대전을 통한 학습
    
    :param num_episodes: 학습 에피소드 수
    """
    # 에이전트 초기화
    white_agent = PPOAgent()
    black_agent = PPOAgent()
    
    # 최종 결과 기록
    white_wins = 0
    black_wins = 0
    draws = 0
    
    for episode in range(num_episodes):
        # 게임 초기화
        board = OthelloBoard()
        current_agent = white_agent  # 흰돌부터 시작
        opponent_agent = black_agent
        
        # 에피소드 진행
        turn_count = 0
        while not board.is_game_over() and turn_count < 100:  # 무한 루프 방지
            # 현재 플레이어의 유효한 이동 확인
            valid_moves = board.get_valid_moves()
            
            logger.debug(f"Turn {turn_count + 1}: {current_agent.__class__.__name__} (Current player: {board.current_player})")
            logger.debug(f"Valid moves: {valid_moves}")
            
            if not valid_moves:
                # 둘 수 있는 수가 없으면 플레이어 교체
                logger.debug(f"No valid moves for {current_agent.__class__.__name__}. Switching players.")
                current_agent, opponent_agent = opponent_agent, current_agent
                board.current_player *= -1
                continue
            
            # 에이전트의 행동 선택
            action = current_agent.choose_action(board.board, valid_moves)
            logger.debug(f"Chosen action: {action}")
            
            # 유효하지 않은 액션일 경우 패배 처리
            if action not in valid_moves:
                logger.warning(f"Invalid move {action} for current player. Treating as defeat.")
                if current_agent == white_agent:
                    black_wins += 1
                else:
                    white_wins += 1
                break
            
            # 보드 상태 저장 (학습을 위해)
            prev_board_state = board.board.copy()
            
            # 행동 수행
            board.make_move(action[0], action[1])
            logger.debug(f"Board after move:\n{board}")
            
            # 게임 종료 확인 및 보상 계산
            if board.is_game_over():
                white_stones, black_stones = board.count_stones()
                logger.debug(f"Game Over - White Stones: {white_stones}, Black Stones: {black_stones}")
                
                if white_stones > black_stones:
                    white_wins += 1
                    current_agent.store_transition(prev_board_state, action, 1, True)
                    opponent_agent.store_transition(prev_board_state, action, 0, True)
                elif black_stones > white_stones:
                    black_wins += 1
                    current_agent.store_transition(prev_board_state, action, 0, True)
                    opponent_agent.store_transition(prev_board_state, action, 1, True)
                else:
                    draws += 1
                    current_agent.store_transition(prev_board_state, action, 0, True)
                    opponent_agent.store_transition(prev_board_state, action, 0, True)
                
                break
            
            # 플레이어 교체
            current_agent, opponent_agent = opponent_agent, current_agent
            board.current_player *= -1
            
            turn_count += 1
        
        # 에피소드 종료 후 에이전트 업데이트
        white_agent.update()
        black_agent.update()
    
    logger.info(f"White Wins: {white_wins}, Black Wins: {black_wins}, Draws: {draws}")
    
    return white_agent, black_agent

def main():
    white_agent, black_agent = train_ppo_agents()
    
    # 학습된 에이전트 저장 (선택적)
    import torch
    torch.save(white_agent.network.state_dict(), 'white_agent.pth')
    torch.save(black_agent.network.state_dict(), 'black_agent.pth')
    logger.info("Training completed and agents saved.")
    
    # 마지막 경기 기록
    log_file = os.path.join('logs', 'final_game.log')
    record_game(white_agent, black_agent, log_file)
    logger.info(f"Final game recorded in {log_file}")

if __name__ == '__main__':
    main()
