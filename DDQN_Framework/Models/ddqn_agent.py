import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from typing import List, Tuple, Dict, Any

from .neural_networks import MultiObjectiveQNetwork, DuelingMultiObjectiveQNetwork
from ..Utils.preference_manager import PreferenceManager
from ..Utils.experience_buffer import MultiObjectiveExperienceBuffer


class DDQNAgent:
    """
    DDQN智能体，支持多目标学习和偏好权重
    实现算法2的核心组件
    """
    
    def __init__(self, state_dim: int, action_dim: int, preference_space: List[np.ndarray], 
                 config: Dict[str, Any], device: str = None):
        """
        初始化DDQN智能体
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            preference_space: 候选偏好空间
            config: 配置参数
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 网络配置
        network_config = config.get('network', {})
        self.hidden_dims = network_config.get('hidden_dims', [256, 128, 64])
        self.learning_rate = network_config.get('learning_rate', 1e-4)
        self.target_update_freq = network_config.get('target_update_freq', 100)
        self.gamma = network_config.get('gamma', 0.99)
        
        # 训练配置
        training_config = config.get('training', {})
        self.batch_size = training_config.get('batch_size', 32)
        self.epsilon_start = training_config.get('epsilon_start', 1.0)
        self.epsilon_final = training_config.get('epsilon_final', 0.1)
        self.epsilon_decay = training_config.get('epsilon_decay', 5000)
        
        # 偏好配置
        preference_config = config.get('preference', {})
        self.preference_manager = PreferenceManager(preference_config)
        
        # 缓冲池配置
        buffer_config = config.get('buffer', {})
        buffer_capacity = buffer_config.get('capacity', 10000)
        self.experience_buffer = MultiObjectiveExperienceBuffer(
            buffer_capacity, state_dim, action_dim
        )
        
        # 网络架构选择
        use_dueling = network_config.get('use_dueling', True)
        if use_dueling:
            self.q_network = DuelingMultiObjectiveQNetwork(
                state_dim, action_dim, self.hidden_dims
            ).to(self.device)
            self.target_q_network = DuelingMultiObjectiveQNetwork(
                state_dim, action_dim, self.hidden_dims
            ).to(self.device)
        else:
            self.q_network = MultiObjectiveQNetwork(
                state_dim, action_dim, self.hidden_dims
            ).to(self.device)
            self.target_q_network = MultiObjectiveQNetwork(
                state_dim, action_dim, self.hidden_dims
            ).to(self.device)
        
        # 初始化目标网络
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # 训练状态
        self.total_steps = 0
        self.update_count = 0
        
        # 性能统计
        self.training_losses = []
        self.episode_rewards = []
        
    def select_action(self, state: np.ndarray, preference: np.ndarray, epsilon: float = None) -> List[int]:
        """
        选择动作（ε-贪心策略）
        Args:
            state: 当前状态
            preference: 偏好权重
            epsilon: 探索率，None表示自动计算
        Returns:
            选择的动作列表
        """
        if epsilon is None:
            epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                     math.exp(-1. * self.total_steps / self.epsilon_decay)
        
        self.total_steps += 1
        
        if random.random() < epsilon:
            # 随机探索
            return [random.randrange(self.action_dim) for _ in range(self.get_vehicle_number())]
        else:
            # 利用策略
            return self._select_greedy_action(state, preference)
    
    def _select_greedy_action(self, state: np.ndarray, preference: np.ndarray) -> List[int]:
        """
        选择贪婪动作
        Args:
            state: 当前状态
            preference: 偏好权重
        Returns:
            贪婪动作列表
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            preference_tensor = torch.FloatTensor(preference).to(self.device)
            
            # 计算多目标Q值
            q_values = self.q_network(state_tensor, preference_tensor)
            
            # 使用偏好权重进行标量化
            scalarized_q_values = self._linear_scalarization(q_values, preference_tensor)
            
            # 选择最大Q值对应的动作
            actions = torch.argmax(scalarized_q_values, dim=1).cpu().numpy()
            
            return actions.tolist()
    
    def _linear_scalarization(self, q_values: torch.Tensor, preference: torch.Tensor) -> torch.Tensor:
        """
        线性标量化函数 f(Vπ, ω) = ωD·VπD + ωE·VπE
        Args:
            q_values: 多目标Q值 (batch_size, num_objectives, action_dim)
            preference: 偏好权重 (num_objectives,)
        Returns:
            标量化Q值 (batch_size, action_dim)
        """
        # 确保偏好是二维张量
        if preference.dim() == 1:
            preference = preference.unsqueeze(0)
        
        # 重复偏好以匹配批次大小
        if q_values.size(0) != preference.size(0):
            preference = preference.expand(q_values.size(0), -1)
        
        # 线性标量化
        scalarized_q = torch.sum(q_values * preference.unsqueeze(-1), dim=1)
        
        return scalarized_q
    
    def train_step(self, batch_data: Tuple) -> float:
        """
        执行一步训练
        Args:
            batch_data: 批次数据 (states, actions, rewards, next_states, dones, preferences)
        Returns:
            训练损失
        """
        states, actions, rewards, next_states, dones, preferences = batch_data
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        preferences = torch.FloatTensor(preferences).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states, preferences)
        
        # 计算目标Q值
        target_q_values = self._compute_target_q_values(next_states, rewards, dones, preferences)
        
        # 计算双偏好损失
        loss = self._compute_dual_preference_loss(
            current_q_values, target_q_values, actions, preferences
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
        
        # 记录损失
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        return loss_value
    
    def _compute_target_q_values(self, next_states: torch.Tensor, rewards: torch.Tensor,
                                dones: torch.Tensor, preferences: torch.Tensor) -> torch.Tensor:
        """
        计算目标Q值
        Args:
            next_states: 下一状态
            rewards: 奖励
            dones: 结束标志
            preferences: 偏好权重
        Returns:
            目标Q值
        """
        with torch.no_grad():
            # Double DQN: 使用主网络选择动作，目标网络评估
            next_q_values = self.q_network(next_states, preferences)
            next_actions = torch.argmax(next_q_values.mean(dim=1), dim=1, keepdim=True)
            
            target_next_q_values = self.target_q_network(next_states, preferences)
            target_next_q = target_next_q_values.gather(1, next_actions.unsqueeze(1).expand(-1, 2, -1))
            
            # 计算目标Q值
            target_q = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1).unsqueeze(-1)) * self.gamma * target_next_q
            
            return target_q
    
    def _compute_dual_preference_loss(self, current_q_values: torch.Tensor, 
                                    target_q_values: torch.Tensor, actions: torch.Tensor,
                                    preferences: torch.Tensor) -> torch.Tensor:
        """
        计算双偏好损失函数
        L = Σ[|Q(si,ai;ωj) - Q̂(si,ai;ωj)| + |Q(si,ai;ωi) - Q̂(si,ai;ωi)|] / 2
        Args:
            current_q_values: 当前Q值
            target_q_values: 目标Q值
            actions: 执行的动作
            preferences: 当前偏好
        Returns:
            损失值
        """
        batch_size = current_q_values.size(0)
        total_loss = 0
        
        for i in range(batch_size):
            # 当前偏好ωj
            current_pref = preferences[i]
            
            # 从遇到偏好中选择ωi
            encountered_pref = self.preference_manager.select_encountered_preference()
            encountered_pref_tensor = torch.FloatTensor(encountered_pref).to(self.device)
            
            # 计算当前偏好的Q值
            current_q = current_q_values[i].gather(0, actions[i].unsqueeze(0).unsqueeze(0).expand(-1, 2))
            target_q = target_q_values[i].gather(0, actions[i].unsqueeze(0).unsqueeze(0).expand(-1, 2))
            
            # 计算遇到偏好的Q值
            state_i = current_q_values[i:i+1]  # 保持批次维度
            encountered_q = self.q_network(state_i, encountered_pref_tensor.unsqueeze(0))
            encountered_q = encountered_q.squeeze(0).gather(0, actions[i].unsqueeze(0).unsqueeze(0).expand(-1, 2))
            
            # 计算遇到偏好的目标Q值
            encountered_target_q = self._compute_target_q_values(
                current_q_values[i:i+1], 
                target_q_values[i:i+1], 
                torch.tensor([False]).to(self.device), 
                encountered_pref_tensor.unsqueeze(0)
            )
            encountered_target_q = encountered_target_q.squeeze(0).gather(0, actions[i].unsqueeze(0).unsqueeze(0).expand(-1, 2))
            
            # 计算损失
            loss_current = torch.abs(current_q - target_q).mean()
            loss_encountered = torch.abs(encountered_q - encountered_target_q).mean()
            
            total_loss += (loss_current + loss_encountered) / 2
        
        return total_loss / batch_size
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def compute_multi_objective_q_values(self, state: np.ndarray, preference: np.ndarray) -> np.ndarray:
        """
        计算多目标Q值
        Args:
            state: 状态
            preference: 偏好权重
        Returns:
            多目标Q值
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            preference_tensor = torch.FloatTensor(preference).to(self.device)
            
            q_values = self.q_network(state_tensor, preference_tensor)
            return q_values.squeeze(0).cpu().numpy()
    
    def store_experience(self, state: np.ndarray, action: List[int], 
                        reward_vector: np.ndarray, next_state: np.ndarray, 
                        done: bool, preference: np.ndarray):
        """
        存储经验到缓冲池
        Args:
            state: 当前状态
            action: 执行的动作
            reward_vector: 多目标奖励向量
            next_state: 下一状态
            done: 是否结束
            preference: 偏好权重
        """
        self.experience_buffer.store(state, action, reward_vector, next_state, done, preference)
        
        # 添加遇到的偏好
        self.preference_manager.add_encountered_preference(preference)
    
    def sample_batch(self) -> Tuple:
        """
        从缓冲池采样批次数据
        Returns:
            批次数据
        """
        return self.experience_buffer.sample_batch_numpy(self.batch_size)
    
    def can_train(self) -> bool:
        """检查是否可以开始训练"""
        return self.experience_buffer.get_buffer_size() >= self.batch_size
    
    def get_vehicle_number(self) -> int:
        """获取车辆数量（从环境适配器获取）"""
        # 这里需要从环境适配器获取，暂时返回默认值
        return 60  # 默认车辆数量
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'buffer_size': self.experience_buffer.get_buffer_size(),
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0.0,
            'encountered_preferences': len(self.preference_manager.get_encountered_preferences())
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dims': self.hidden_dims,
                'learning_rate': self.learning_rate
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 