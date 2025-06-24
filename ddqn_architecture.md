# 基于改进DRL的动态权重算法代码架构

## 1. 整体架构概述

### 1.1 核心模块划分
```
DDQN_Framework/
├── Models/
│   ├── ddqn_agent.py          # DDQN智能体主类
│   ├── neural_networks.py     # 神经网络结构定义
│   └── trajectory_predictor.py # TPPT轨迹预测模块
├── Utils/
│   ├── multi_objective.py     # 多目标处理工具
│   ├── preference_manager.py  # 偏好管理器
│   └── experience_buffer.py   # 经验回放缓冲池
├── Training/
│   ├── trainer.py            # 训练主控制器
│   └── evaluation.py         # 评估模块
└── main.py                   # 主程序入口
```

## 2. 详细模块设计

### 2.1 DDQN智能体模块 (`Models/ddqn_agent.py`)

#### 2.1.1 类结构
```python
class DDQNAgent:
    def __init__(self, state_dim, action_dim, preference_space, config)
    def select_action(self, state, preference, epsilon)
    def train_step(self, batch_data)
    def update_target_network(self)
    def compute_multi_objective_q_values(self, state, preference)
    def non_dominated_sorting(self, encountered_preferences)
```

#### 2.1.2 关键属性
- `q_network`: 主Q网络
- `target_q_network`: 目标Q网络  
- `preference_space`: 候选偏好空间Ω
- `encountered_preferences`: 遇到的偏好集合W
- `experience_buffer`: 经验回放缓冲池
- `trajectory_predictor`: TPPT轨迹预测器

#### 2.1.3 核心方法实现要点
- **状态编码**: 整合TPPT预测结果到状态表示
- **多目标Q值计算**: 输出延迟和能耗两个维度的Q值向量
- **动作选择**: ε-贪心策略结合偏好权重
- **损失函数**: 实现论文中的双偏好损失函数

### 2.2 神经网络结构 (`Models/neural_networks.py`)

#### 2.2.1 Q网络架构
```python
class MultiObjectiveQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, num_objectives=2)
    def forward(self, state, preference)
    def _build_shared_layers(self)
    def _build_objective_heads(self)
```

#### 2.2.2 网络设计要点
- **共享特征提取层**: 处理状态信息的公共特征
- **多目标输出头**: 分别输出延迟Q值和能耗Q值
- **偏好融合机制**: 将偏好权重融入网络计算过程

### 2.3 轨迹预测模块 (`Models/trajectory_predictor.py`)

#### 2.3.1 TPPT算法实现
```python
class TrajectoryPredictor:
    def __init__(self, rsu_locations, config)
    def predict_next_rsu(self, current_trajectory, mode='Predict')
    def update_model(self, feedback_data, mode='Update')
    def build_tpt_tree(self, historical_data)
    def match_prefix(self, trajectory, tpt_tree)
    def find_frequent_patterns(self, prefix, item_table, frequency_table)
```

#### 2.3.2 数据结构设计
- **T-模式树(TPT)**: 存储轨迹模式的树形结构
- **项目表(IT)**: 记录轨迹项目和支持度
- **频率表(FR)**: 存储频繁模式及其评分

### 2.4 多目标处理工具 (`Utils/multi_objective.py`)

#### 2.4.1 核心功能
```python
class MultiObjectiveUtils:
    @staticmethod
    def linear_scalarization(q_vector, preference)
    @staticmethod
    def non_dominated_sorting(preferences)
    @staticmethod
    def compute_reward_vector(delay, energy)
    @staticmethod
    def normalize_objectives(objectives)
```

#### 2.4.2 实现要点
- **标量化函数**: f(Vπ, ω) = ωD·VπD + ωE·VπE
- **非支配排序**: 从遇到偏好中选择合适偏好
- **奖励向量计算**: 分别计算延迟奖励和能耗奖励

### 2.5 偏好管理器 (`Utils/preference_manager.py`)

#### 2.5.1 偏好空间管理
```python
class PreferenceManager:
    def __init__(self, preference_space_config)
    def sample_preference(self)
    def add_encountered_preference(self, preference)
    def select_encountered_preference(self, non_dominated_sorting=True)
    def validate_preference(self, preference)
```

#### 2.5.2 关键功能
- **偏好采样**: 从候选偏好空间Ω中采样
- **偏好验证**: 确保ωD + ωE = 1且ωD, ωE ≥ 0
- **遇到偏好管理**: 维护和选择历史遇到的偏好

### 2.6 经验回放缓冲池 (`Utils/experience_buffer.py`)

#### 2.6.1 缓冲池设计
```python
class ExperienceBuffer:
    def __init__(self, capacity, state_dim, action_dim)
    def store(self, state, action, reward_vector, next_state, done, preference)
    def sample_batch(self, batch_size)
    def get_buffer_size(self)
    def clear(self)
```

#### 2.6.2 存储结构
- **经验元组**: (st, at, rt, st+1, done, ωt)
- **多目标奖励**: rt = [rtD, rtE] (延迟奖励，能耗奖励)
- **偏好信息**: 存储对应的偏好权重

## 3. 训练流程设计

### 3.1 主训练循环 (`Training/trainer.py`)

#### 3.1.1 训练器结构
```python
class DDQNTrainer:
    def __init__(self, env, agent, config)
    def train(self, max_episodes)
    def train_episode(self, episode_idx)
    def collect_experience(self, horizon_steps)
    def update_networks(self, batch_data)
    def evaluate_performance(self, eval_episodes)
```

#### 3.1.2 训练步骤实现
1. **偏好采样**: 从Ω中采样当前偏好ωj
2. **经验收集**: 收集Thorizon步经验数据
3. **批次训练**: 从经验池采样进行网络更新
4. **目标网络同步**: 每N步同步一次目标网络参数

### 3.2 状态空间扩展

#### 3.2.1 增强状态表示
```python
def build_enhanced_state(self, vehicle_states, tppt_predictions):
    """
    构建包含TPPT预测的增强状态空间
    st = (Lv(t), PredTPPT(t), ChSt, RSUS(t), TaskC(t))
    """
    enhanced_state = []
    for i, vehicle in enumerate(vehicles):
        # 车辆当前RSU连接
        current_rsu = vehicle_states[i]['current_rsu']
        # TPPT预测的下一个RSU
        predicted_rsu = tppt_predictions[i]
        # 信道状态信息
        channel_state = vehicle_states[i]['channel_state']
        # 候选RSU资源状态
        rsu_resources = vehicle_states[i]['rsu_resources']
        # 当前任务特征
        task_features = vehicle_states[i]['task_features']
        
        vehicle_state = np.concatenate([
            [current_rsu, predicted_rsu],
            channel_state,
            rsu_resources,
            task_features
        ])
        enhanced_state.extend(vehicle_state)
    
    return np.array(enhanced_state)
```

### 3.3 奖励函数设计

#### 3.3.1 多目标奖励计算
```python
def compute_multi_objective_reward(self, vehicles, actions):
    """
    计算多目标奖励向量 rt = [rtD, rtE]
    """
    delay_rewards = []
    energy_rewards = []
    
    for i, vehicle in enumerate(vehicles):
        action = actions[i]  # 0: 本地执行, 1: 卸载
        
        if action == 0:  # 本地执行
            delay = self.compute_local_delay(vehicle)
            energy = self.compute_local_energy(vehicle)
        else:  # 卸载执行
            delay = self.compute_offload_delay(vehicle)
            energy = self.compute_offload_energy(vehicle)
        
        # 延迟奖励（负延迟，最小化延迟）
        delay_reward = -delay
        # 能耗奖励（负能耗，最小化能耗）
        energy_reward = -energy
        
        delay_rewards.append(delay_reward)
        energy_rewards.append(energy_reward)
    
    return np.array([sum(delay_rewards), sum(energy_rewards)])
```

## 4. 关键算法实现要点

### 4.1 算法2核心逻辑

#### 4.1.1 主训练循环伪代码映射
```python
def algorithm_2_implementation(self):
    """
    算法2：基于改进DRL的动态权重算法的具体实现流程
    """
    # Step 1-3: 初始化
    self.initialize_components()
    
    # Step 4: 主训练循环
    for episode in range(self.max_episodes):
        # Step 5: 采样当前偏好
        current_preference = self.preference_manager.sample_preference()
        
        # Step 6: 观察初始状态
        state = self.env.reset()
        
        # Step 7-11: 收集经验
        for t in range(self.horizon_steps):
            # Step 8: 选择动作(ε-贪心)
            action = self.agent.select_action(state, current_preference, self.epsilon)
            
            # Step 9: 执行动作，获得奖励和下一状态
            next_state, reward_vector, done = self.env.step(action)
            
            # Step 10: 存储经验到缓冲池
            self.experience_buffer.store(state, action, reward_vector, next_state, done, current_preference)
            
            state = next_state
            if done:
                break
        
        # Step 12-22: 批次训练
        if self.experience_buffer.size() >= self.batch_size:
            batch_data = self.experience_buffer.sample_batch(self.batch_size)
            self.train_batch(batch_data)
        
        # Step 21: 定期同步目标网络
        if episode % self.target_update_freq == 0:
            self.agent.update_target_network()
```

### 4.2 损失函数实现

#### 4.2.1 双偏好损失函数
```python
def compute_loss(self, batch_data):
    """
    实现论文中的损失函数：
    L = Σ[|Q(si,ai;ωj) - Q̂(si,ai;ωj)| + |Q(si,ai;ωi) - Q̂(si,ai;ωi)|] / 2
    """
    states, actions, rewards, next_states, dones, preferences = batch_data
    
    total_loss = 0
    batch_size = len(states)
    
    for i in range(batch_size):
        # 当前偏好ωj
        current_pref = preferences[i]
        
        # 从遇到偏好中选择ωi
        encountered_pref = self.preference_manager.select_encountered_preference()
        
        # 计算Q值
        q_current = self.agent.compute_q_value(states[i], actions[i], current_pref)
        q_encountered = self.agent.compute_q_value(states[i], actions[i], encountered_pref)
        
        # 计算目标Q值
        if dones[i]:
            target_current = rewards[i]
            target_encountered = rewards[i]
        else:
            target_current = rewards[i] + self.gamma * self.compute_target_q(next_states[i], current_pref)
            target_encountered = rewards[i] + self.gamma * self.compute_target_q(next_states[i], encountered_pref)
        
        # 计算损失
        loss_current = abs(q_current - target_current)
        loss_encountered = abs(q_encountered - target_encountered)
        
        total_loss += (loss_current + loss_encountered) / 2
    
    return total_loss / batch_size
```

## 5. 集成现有环境

### 5.1 环境适配器设计
```python
class VECEnvironmentAdapter:
    """
    适配现有环境以支持DDQN算法
    """
    def __init__(self, original_env, trajectory_predictor)
    def reset(self)
    def step(self, actions)
    def get_enhanced_state(self)
    def compute_multi_objective_rewards(self, actions, results)
```

### 5.2 状态空间映射
- 利用现有的`get_state_normalize()`方法
- 集成TPPT预测结果到状态表示
- 保持与现有车辆和基站模型的兼容性

## 6. 配置参数管理

### 6.1 配置文件结构
```python
DDQN_CONFIG = {
    'network': {
        'hidden_dims': [256, 128, 64],
        'learning_rate': 1e-4,
        'target_update_freq': 100
    },
    'training': {
        'max_episodes': 5000,
        'horizon_steps': 200,
        'batch_size': 32,
        'epsilon_decay': 0.995
    },
    'preference': {
        'preference_space_size': 11,  # ω ∈ {0.0, 0.1, ..., 1.0}
        'non_dominated_sorting': True
    },
    'tppt': {
        'min_support': 0.1,
        'window_size': 10
    }
}
```

这个架构设计确保了算法2的完整实现，同时保持了与现有代码的良好集成。每个模块都有明确的职责分工，便于开发、测试和维护。