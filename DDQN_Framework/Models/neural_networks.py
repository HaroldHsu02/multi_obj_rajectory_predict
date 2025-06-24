import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiObjectiveQNetwork(nn.Module):
    """
    多目标Q网络，支持延迟和能耗两个目标
    实现共享特征提取层、偏好融合机制和多目标输出头
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 128, 64], num_objectives=2):
        super(MultiObjectiveQNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.num_objectives = num_objectives
        
        # 构建共享特征提取层
        self.shared_layers = self._build_shared_layers(state_dim, hidden_dims)
        
        # 构建多目标输出头
        self.objective_heads = self._build_objective_heads(hidden_dims, action_dim, num_objectives)
        
        # 偏好融合层
        self.preference_fusion = nn.Linear(hidden_dims[-1] + num_objectives, hidden_dims[-1])
        
    def _build_shared_layers(self, state_dim, hidden_dims):
        """构建共享特征提取层"""
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
            
        return nn.Sequential(*layers)
    
    def _build_objective_heads(self, hidden_dims, action_dim, num_objectives):
        """构建多目标输出头"""
        heads = nn.ModuleList()
        for _ in range(num_objectives):
            head = nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1] // 2, action_dim)
            )
            heads.append(head)
        return heads
    
    def _fuse_preference(self, features, preference):
        """将偏好权重融合到特征中"""
        # 确保偏好是二维张量
        if preference.dim() == 1:
            preference = preference.unsqueeze(0)
        
        # 重复偏好以匹配批次大小
        if features.size(0) != preference.size(0):
            preference = preference.expand(features.size(0), -1)
        
        # 连接特征和偏好
        fused_features = torch.cat([features, preference], dim=1)
        return self.preference_fusion(fused_features)
    
    def forward(self, state, preference):
        """
        前向传播
        Args:
            state: 状态张量 (batch_size, state_dim)
            preference: 偏好权重 (batch_size, num_objectives) 或 (num_objectives,)
        Returns:
            q_values: 多目标Q值 (batch_size, num_objectives, action_dim)
        """
        # 通过共享层提取特征
        features = self.shared_layers(state)
        
        # 融合偏好信息
        fused_features = self._fuse_preference(features, preference)
        
        # 通过多目标输出头计算Q值
        q_values = []
        for head in self.objective_heads:
            q_value = head(fused_features)  # (batch_size, action_dim)
            q_values.append(q_value)
        
        # 堆叠为多目标Q值张量
        q_values = torch.stack(q_values, dim=1)  # (batch_size, num_objectives, action_dim)
        
        return q_values


class DuelingMultiObjectiveQNetwork(nn.Module):
    """
    基于Dueling架构的多目标Q网络
    结合Value和Advantage分支的优势
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 128, 64], num_objectives=2):
        super(DuelingMultiObjectiveQNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.num_objectives = num_objectives
        
        # 共享特征提取层
        self.shared_layers = self._build_shared_layers(state_dim, hidden_dims)
        
        # Value分支
        self.value_heads = self._build_value_heads(hidden_dims, num_objectives)
        
        # Advantage分支
        self.advantage_heads = self._build_advantage_heads(hidden_dims, action_dim, num_objectives)
        
        # 偏好融合层
        self.preference_fusion = nn.Linear(hidden_dims[-1] + num_objectives, hidden_dims[-1])
        
    def _build_shared_layers(self, state_dim, hidden_dims):
        """构建共享特征提取层"""
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
            
        return nn.Sequential(*layers)
    
    def _build_value_heads(self, hidden_dims, num_objectives):
        """构建Value分支"""
        heads = nn.ModuleList()
        for _ in range(num_objectives):
            head = nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1] // 2, 1)
            )
            heads.append(head)
        return heads
    
    def _build_advantage_heads(self, hidden_dims, action_dim, num_objectives):
        """构建Advantage分支"""
        heads = nn.ModuleList()
        for _ in range(num_objectives):
            head = nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1] // 2, action_dim)
            )
            heads.append(head)
        return heads
    
    def _fuse_preference(self, features, preference):
        """将偏好权重融合到特征中"""
        if preference.dim() == 1:
            preference = preference.unsqueeze(0)
        
        if features.size(0) != preference.size(0):
            preference = preference.expand(features.size(0), -1)
        
        fused_features = torch.cat([features, preference], dim=1)
        return self.preference_fusion(fused_features)
    
    def forward(self, state, preference):
        """
        前向传播
        Args:
            state: 状态张量 (batch_size, state_dim)
            preference: 偏好权重 (batch_size, num_objectives) 或 (num_objectives,)
        Returns:
            q_values: 多目标Q值 (batch_size, num_objectives, action_dim)
        """
        # 通过共享层提取特征
        features = self.shared_layers(state)
        
        # 融合偏好信息
        fused_features = self._fuse_preference(features, preference)
        
        # 计算Value和Advantage
        values = []
        advantages = []
        
        for i in range(self.num_objectives):
            value = self.value_heads[i](fused_features)  # (batch_size, 1)
            advantage = self.advantage_heads[i](fused_features)  # (batch_size, action_dim)
            
            values.append(value)
            advantages.append(advantage)
        
        # 组合Value和Advantage
        q_values = []
        for i in range(self.num_objectives):
            # 去均值操作
            advantage_mean = advantages[i].mean(dim=1, keepdim=True)
            q_value = values[i] + (advantages[i] - advantage_mean)
            q_values.append(q_value)
        
        # 堆叠为多目标Q值张量
        q_values = torch.stack(q_values, dim=1)  # (batch_size, num_objectives, action_dim)
        
        return q_values 