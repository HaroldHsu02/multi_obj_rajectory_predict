import numpy as np
import random
from typing import List, Tuple, Optional
from collections import deque


class MultiObjectiveExperienceBuffer:
    """
    多目标经验回放缓冲池
    存储格式：(st, at, rt, st+1, done, ωt)
    其中rt是多目标奖励向量 [rtD, rtE]
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        """
        初始化经验缓冲池
        Args:
            capacity: 缓冲池容量
            state_dim: 状态维度
            action_dim: 动作维度
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 使用deque实现循环缓冲
        self.buffer = deque(maxlen=capacity)
        
        # 经验统计
        self.total_experiences = 0
        self.valid_experiences = 0
        
    def store(self, state: np.ndarray, action: np.ndarray, 
              reward_vector: np.ndarray, next_state: np.ndarray, 
              done: bool, preference: np.ndarray) -> bool:
        """
        存储一个经验样本
        Args:
            state: 当前状态 (state_dim,)
            action: 执行的动作 (action_dim,) 或标量
            reward_vector: 多目标奖励向量 [rtD, rtE]
            next_state: 下一状态 (state_dim,)
            done: 是否结束
            preference: 偏好权重 [ωD, ωE]
        Returns:
            是否成功存储
        """
        # 验证经验数据
        if not self._validate_experience(state, action, reward_vector, next_state, done, preference):
            return False
        
        # 确保数据类型和形状正确
        state = np.array(state, dtype=np.float32)
        if np.isscalar(action):
            action = np.array([action], dtype=np.int64)
        else:
            action = np.array(action, dtype=np.int64)
        reward_vector = np.array(reward_vector, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        preference = np.array(preference, dtype=np.float32)
        
        # 存储经验
        experience = (state, action, reward_vector, next_state, done, preference)
        self.buffer.append(experience)
        
        self.total_experiences += 1
        self.valid_experiences += 1
        
        return True
    
    def sample_batch(self, batch_size: int) -> Tuple[List, List, List, List, List, List]:
        """
        采样批次数据
        Args:
            batch_size: 批次大小
        Returns:
            批次数据元组 (states, actions, rewards, next_states, dones, preferences)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"缓冲池中经验数量({len(self.buffer)})少于批次大小({batch_size})")
        
        # 随机采样
        batch = random.sample(self.buffer, batch_size)
        
        # 解包批次数据
        states, actions, rewards, next_states, dones, preferences = zip(*batch)
        
        return states, actions, rewards, next_states, dones, preferences
    
    def sample_batch_numpy(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                          np.ndarray, np.ndarray, np.ndarray]:
        """
        采样批次数据并转换为numpy数组
        Args:
            batch_size: 批次大小
        Returns:
            numpy格式的批次数据
        """
        states, actions, rewards, next_states, dones, preferences = self.sample_batch(batch_size)
        
        # 转换为numpy数组
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)
        preferences = np.array(preferences, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, preferences
    
    def get_buffer_size(self) -> int:
        """获取当前缓冲池大小"""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """检查缓冲池是否已满"""
        return len(self.buffer) >= self.capacity
    
    def clear(self):
        """清空缓冲池"""
        self.buffer.clear()
        self.total_experiences = 0
        self.valid_experiences = 0
    
    def _validate_experience(self, state: np.ndarray, action: np.ndarray, 
                           reward_vector: np.ndarray, next_state: np.ndarray, 
                           done: bool, preference: np.ndarray) -> bool:
        """
        验证经验数据
        Args:
            state: 当前状态
            action: 执行的动作
            reward_vector: 多目标奖励向量
            next_state: 下一状态
            done: 是否结束
            preference: 偏好权重
        Returns:
            是否有效
        """
        try:
            # 检查状态
            if not isinstance(state, (np.ndarray, list)) or len(state) != self.state_dim:
                return False
            
            # 检查动作
            if np.isscalar(action):
                if not isinstance(action, (int, np.integer)) or action < 0 or action >= self.action_dim:
                    return False
            else:
                if not isinstance(action, (np.ndarray, list)) or len(action) != self.action_dim:
                    return False
            
            # 检查奖励向量
            if not isinstance(reward_vector, (np.ndarray, list)) or len(reward_vector) != 2:
                return False
            
            # 检查下一状态
            if not isinstance(next_state, (np.ndarray, list)) or len(next_state) != self.state_dim:
                return False
            
            # 检查done标志
            if not isinstance(done, bool):
                return False
            
            # 检查偏好权重
            if not isinstance(preference, (np.ndarray, list)) or len(preference) != 2:
                return False
            
            # 验证偏好权重和为1
            pref_array = np.array(preference)
            if abs(pref_array[0] + pref_array[1] - 1.0) > 1e-6:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_statistics(self) -> dict:
        """获取缓冲池统计信息"""
        if len(self.buffer) == 0:
            return {
                'buffer_size': 0,
                'total_experiences': self.total_experiences,
                'valid_experiences': self.valid_experiences,
                'validity_rate': 0.0,
                'avg_delay_reward': 0.0,
                'avg_energy_reward': 0.0
            }
        
        # 计算平均奖励
        delay_rewards = []
        energy_rewards = []
        
        for _, _, reward_vector, _, _, _ in self.buffer:
            delay_rewards.append(reward_vector[0])
            energy_rewards.append(reward_vector[1])
        
        return {
            'buffer_size': len(self.buffer),
            'total_experiences': self.total_experiences,
            'valid_experiences': self.valid_experiences,
            'validity_rate': self.valid_experiences / max(self.total_experiences, 1),
            'avg_delay_reward': np.mean(delay_rewards),
            'avg_energy_reward': np.mean(energy_rewards)
        }
    
    def sample_by_preference(self, target_preference: np.ndarray, batch_size: int, 
                           tolerance: float = 0.1) -> Optional[Tuple]:
        """
        根据偏好采样经验
        Args:
            target_preference: 目标偏好
            batch_size: 批次大小
            tolerance: 偏好匹配容差
        Returns:
            匹配的经验批次，如果没有匹配则返回None
        """
        matching_experiences = []
        
        for experience in self.buffer:
            _, _, _, _, _, pref = experience
            if self._preference_match(pref, target_preference, tolerance):
                matching_experiences.append(experience)
        
        if len(matching_experiences) < batch_size:
            return None
        
        # 随机采样匹配的经验
        batch = random.sample(matching_experiences, batch_size)
        states, actions, rewards, next_states, dones, preferences = zip(*batch)
        
        return states, actions, rewards, next_states, dones, preferences
    
    def _preference_match(self, pref1: np.ndarray, pref2: np.ndarray, tolerance: float) -> bool:
        """
        检查两个偏好是否匹配
        Args:
            pref1: 偏好1
            pref2: 偏好2
            tolerance: 容差
        Returns:
            是否匹配
        """
        return np.allclose(pref1, pref2, atol=tolerance)
    
    def get_preference_distribution(self) -> dict:
        """
        获取偏好分布统计
        Returns:
            偏好分布字典
        """
        if len(self.buffer) == 0:
            return {}
        
        preference_counts = {}
        
        for _, _, _, _, _, pref in self.buffer:
            pref_key = tuple(np.round(pref, 2))  # 四舍五入到小数点后2位
            preference_counts[pref_key] = preference_counts.get(pref_key, 0) + 1
        
        return preference_counts 