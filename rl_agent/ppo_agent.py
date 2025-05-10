import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

logger = logging.getLogger(__name__)

class OthelloPPONetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Othello PPO 신경망 구조
        
        :param input_dim: 입력 차원 (보드 상태)
        :param output_dim: 출력 차원 (가능한 행동)
        """
        super(OthelloPPONetwork, self).__init__()
        
        # 정책 네트워크
        self.policy_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.LogSoftmax(dim=-1)
        )
        
        # 가치 네트워크
        self.value_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # 가중치 초기화
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        
        self.policy_network.apply(init_weights)
        self.value_network.apply(init_weights)
    
    def forward(self, state):
        """
        신경망을 통해 정책과 가치 계산
        
        :param state: 입력 상태 텐서
        :return: 정책 및 가치 출력
        """
        state = state.float()
        policy_logits = self.policy_network(state)
        policy = F.softmax(policy_logits, dim=-1)
        value = self.value_network(state)
        
        return policy_logits, value

class PPOAgent:
    def __init__(self, board_size=8, learning_rate=1e-4, clip_range=0.2):
        """
        PPO 에이전트 초기화
        
        :param board_size: 보드 크기
        :param learning_rate: 학습률
        :param clip_range: PPO 클리핑 범위
        """
        # CUDA 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'학습에 사용될 디바이스: {self.device}')
        
        self.board_size = board_size
        self.input_dim = board_size * board_size
        self.output_dim = board_size * board_size
        
        # 신경망 및 최적화기 초기화
        self.network = OthelloPPONetwork(self.input_dim, self.output_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # PPO 하이퍼파라미터
        self.clip_range = clip_range
        self.epochs = 3
        self.batch_size = 32
        self.mini_batch_size = 16
        self.gamma = 0.99
        
        # 학습 데이터 저장소
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.old_log_probs = []
    
    def preprocess_state(self, board_state):
        """
        보드 상태를 신경망 입력 형식으로 변환
        
        :param board_state: 현재 보드 상태
        :return: 전처리된 상태 텐서
        """
        state_flat = board_state.flatten().astype(np.float32)
        return torch.tensor(state_flat).to(self.device).unsqueeze(0)
    
    def choose_action(self, board_state, valid_moves, learning_mode=True):
        """
        PPO 정책에 따라 행동 선택
        
        :param board_state: 현재 보드 상태
        :param valid_moves: 유효한 이동 위치들
        :param learning_mode: 학습 모드 여부
        :return: 선택된 행동
        """
        # 8x8 보드의 모든 위치 생성
        all_moves = [(x, y) for x in range(self.board_size) for y in range(self.board_size)]
        
        if not valid_moves:
            print(f"No valid moves available for current player")
            return None
        
        # 상태 전처리
        state = self.preprocess_state(board_state)
        
        # 정책 및 가치 계산
        with torch.no_grad():
            policy_logits, _ = self.network(state)
        
        # 탐험 vs 활용
        if learning_mode and random.random() < 0.1:
            # 탐험시 유효한 이동만 선택
            action = random.choice(valid_moves)
            logger.debug(f"Exploration: Randomly selected action {action}")
        else:
            # 마스킹된 정책에서 행동 선택
            action_probs = F.softmax(policy_logits, dim=-1).squeeze()
            
            # 모든 위치에 대한 확률 계산
            all_action_probs = [action_probs[x * self.board_size + y].item() for x, y in all_moves]
            
            # 유효한 이동만 필터링
            valid_action_probs = [all_action_probs[all_moves.index(move)] for move in valid_moves]
            
            print(f"Valid moves: {valid_moves}")
            print(f"Action probabilities: {valid_action_probs}")
            
            # NaN 방지 및 유효성 검사
            if any(np.isnan(p) for p in valid_action_probs) or sum(valid_action_probs) == 0:
                action = random.choice(valid_moves)
                print(f"Invalid probabilities. Randomly selected action {action}")
            else:
                valid_action_probs_norm = [p / sum(valid_action_probs) for p in valid_action_probs]
                action_idx = np.random.choice(len(valid_moves), p=valid_action_probs_norm)
                action = valid_moves[action_idx]
                print(f"Policy-based action selection: {action}")
        
        # 학습을 위한 로그 확률 저장
        action_flat_idx = action[0] * self.board_size + action[1]
        log_prob = policy_logits[0, action_flat_idx]
        self.old_log_probs.append(log_prob.to('cpu'))
        
        return action
    
    def store_transition(self, state, action, reward, done):
        """
        학습을 위한 상태 전이
        
        :param state: 보드 상태
        :param action: 선택한 행동
        :param reward: 보상
        :param done: 에피소드 여부
        """
        # 텐서 변환 및 복사
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.long)
        
        self.states.append(state_tensor)
        self.actions.append(action_tensor)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self):
        """
        PPO 업데이트 스테이지
        """
        if len(self.states) < self.batch_size:
            return
        
        # 배치 준비 및 가치 계산
        batch_states = torch.stack(self.states).to(self.device)
        batch_actions = torch.stack(self.actions).to(self.device)
        batch_rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        batch_dones = torch.tensor(self.dones, dtype=torch.bool).to(self.device)
        
        # 할인판 계산
        returns = []
        advantages = []
        R = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 새로운 데이터로 네트워크 업데이트
        for _ in range(self.epochs):
            # 네트워크 새로 실행
            new_policy_logits, new_values = self.network(batch_states)
            new_policies = F.softmax(new_policy_logits, dim=-1)
            
            # 각 에이전의 로그 확률 계산
            batch_actions_flat = batch_actions[:, 0] * self.board_size + batch_actions[:, 1]
            new_log_probs = torch.log(new_policies.gather(1, batch_actions_flat.unsqueeze(1)).squeeze(1))
            
            # 진짜 비율 계산
            ratios = torch.exp(new_log_probs - torch.tensor(self.old_log_probs, dtype=torch.float32).to(self.device))
            
            # 서블로스 로스 계산
            surr1 = ratios * returns
            surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * returns
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 가치 네트워크 손실
            value_loss = F.mse_loss(new_values.squeeze(), returns)
            
            # 엔트로피 입연률 조정
            entropy_loss = -torch.mean(new_policies * torch.log(new_policies + 1e-10))
            
            # 총 손실 계산
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
            
            # 배치 업데이트
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # 버퍼 지우기
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.old_log_probs.clear()
