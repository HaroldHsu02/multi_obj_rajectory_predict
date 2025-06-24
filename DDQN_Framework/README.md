# DDQN Framework - 基于改进DRL的动态权重算法

## 概述

DDQN Framework 是一个基于改进深度强化学习（DRL）的动态权重算法实现，专门用于车联网环境中的多目标优化问题。该框架实现了论文中提出的算法2，支持延迟和能耗两个目标的动态权重优化。

## 主要特性

- **多目标学习**: 同时优化延迟和能耗两个目标
- **动态权重**: 支持偏好权重的动态调整
- **TPPT集成**: 集成轨迹预测树算法增强状态表示
- **双偏好损失**: 实现论文中的双偏好损失函数
- **Dueling架构**: 支持Dueling DQN网络架构
- **模块化设计**: 清晰的模块分离，易于扩展和维护

## 目录结构

```
DDQN_Framework/
├── Models/                    # 模型定义
│   ├── ddqn_agent.py         # DDQN智能体
│   ├── neural_networks.py    # 神经网络架构
│   └── __init__.py
├── Utils/                    # 工具类
│   ├── preference_manager.py # 偏好管理器
│   ├── experience_buffer.py  # 经验缓冲池
│   └── __init__.py
├── Adapters/                 # 适配器
│   ├── vec_environment_adapter.py # 环境适配器
│   └── __init__.py
├── Training/                 # 训练模块
│   ├── trainer.py           # 训练器
│   └── __init__.py
├── config.py                # 配置文件
├── main.py                  # 主程序入口
├── example.py               # 使用示例
└── README.md               # 本文档
```

## 安装要求

### 系统要求
- Python 3.7+
- PyTorch 1.8+
- NumPy
- 其他依赖见项目根目录的requirements.txt

### 安装步骤

1. 克隆项目到本地
```bash
git clone <repository_url>
cd multi_obj
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 验证安装
```bash
cd DDQN_Framework
python example.py
```

## 快速开始

### 基本使用

```python
from DDQN_Framework import (
    VECEnvironmentAdapter, DDQNAgent, DDQNTrainer,
    get_config
)

# 1. 获取配置
config = get_config('default')

# 2. 设置环境
from Environments.env import Env
original_env = Env()
env_adapter = VECEnvironmentAdapter(original_env)

# 3. 设置智能体
agent = DDQNAgent(
    state_dim=env_adapter.get_state_dim(),
    action_dim=env_adapter.get_action_dim(),
    preference_space=preference_space,
    config=config
)

# 4. 设置训练器
trainer = DDQNTrainer(env_adapter, agent, config)

# 5. 开始训练
training_result = trainer.train(max_episodes=1000)
```

### 命令行使用

```bash
# 使用默认配置训练
python main.py

# 使用快速训练配置
python main.py --config fast_training --episodes 1000

# 只进行评估
python main.py --no_training --eval_episodes 100

# 加载检查点继续训练
python main.py --load_checkpoint 500
```

## 配置说明

### 配置选项

框架提供了多种预定义配置：

- **default**: 标准配置，适合一般训练
- **fast_training**: 快速训练配置，适合调试和快速验证
- **high_precision**: 高精度配置，适合最终训练
- **delay_focused**: 延迟优先配置
- **energy_focused**: 能耗优先配置

### 自定义配置

```python
from DDQN_Framework.config import get_config, merge_config

# 获取基础配置
base_config = get_config('default')

# 自定义配置
custom_config = {
    'training': {
        'max_episodes': 2000,
        'batch_size': 64
    },
    'network': {
        'learning_rate': 5e-5
    }
}

# 合并配置
config = merge_config(base_config, custom_config)
```

### 配置参数说明

#### 网络配置 (network)
- `hidden_dims`: 隐藏层维度列表
- `learning_rate`: 学习率
- `target_update_freq`: 目标网络更新频率
- `gamma`: 折扣因子
- `use_dueling`: 是否使用Dueling架构

#### 训练配置 (training)
- `max_episodes`: 最大训练轮数
- `horizon_steps`: 每轮最大步数
- `batch_size`: 批次大小
- `epsilon_start/final/decay`: ε-贪心策略参数

#### 偏好配置 (preference)
- `preference_space_size`: 偏好空间大小
- `non_dominated_sorting`: 是否使用非支配排序
- `validation_enabled`: 是否启用偏好验证

## 核心组件

### DDQNAgent

DDQN智能体是框架的核心组件，负责：
- 多目标Q值计算
- 动作选择（ε-贪心策略）
- 网络训练和更新
- 经验管理

```python
agent = DDQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    preference_space=preference_space,
    config=config
)

# 选择动作
action = agent.select_action(state, preference)

# 训练一步
loss = agent.train_step(batch_data)
```

### PreferenceManager

偏好管理器负责：
- 候选偏好空间管理
- 偏好采样和验证
- 非支配排序
- 遇到偏好记录

```python
preference_manager = PreferenceManager(config)

# 采样偏好
preference = preference_manager.sample_preference()

# 添加遇到偏好
preference_manager.add_encountered_preference(preference, performance)
```

### MultiObjectiveExperienceBuffer

多目标经验缓冲池支持：
- 多目标奖励存储
- 偏好信息记录
- 批次采样
- 统计信息计算

```python
buffer = MultiObjectiveExperienceBuffer(capacity, state_dim, action_dim)

# 存储经验
buffer.store(state, action, reward_vector, next_state, done, preference)

# 采样批次
batch = buffer.sample_batch_numpy(batch_size)
```

## 算法实现

### 算法2核心流程

1. **初始化**: 创建Q网络、目标网络、经验缓冲池
2. **偏好采样**: 从候选偏好空间Ω中采样当前偏好ωj
3. **经验收集**: 使用ε-贪心策略收集经验
4. **批次训练**: 实现双偏好损失函数
5. **网络更新**: 定期同步目标网络

### 双偏好损失函数

```
L = Σ[|Q(si,ai;ωj) - Q̂(si,ai;ωj)| + |Q(si,ai;ωi) - Q̂(si,ai;ωi)|] / 2
```

其中：
- ωj: 当前偏好
- ωi: 从遇到偏好中选择的偏好
- Q: 当前Q值
- Q̂: 目标Q值

## 性能评估

### 评估指标

- **平均奖励**: 标量化后的平均奖励
- **延迟奖励**: 平均延迟性能
- **能耗奖励**: 平均能耗性能
- **偏好分布**: 不同偏好的使用情况

### 评估方法

```python
# 运行评估
eval_result = trainer.evaluate_performance(eval_episodes=50)

# 查看结果
print(f"平均奖励: {eval_result['avg_reward']:.3f}")
print(f"延迟奖励: {eval_result['avg_delay_reward']:.3f}")
print(f"能耗奖励: {eval_result['avg_energy_reward']:.3f}")
```

## 结果保存

训练结果会自动保存到 `results/` 目录：

```
results/ddqn_YYYYMMDD_HHMMSS/
├── final_model.pth          # 最终模型
├── training_summary.pkl     # 训练摘要
└── config.json             # 配置文件
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减少批次大小
   - 减少缓冲池容量
   - 使用更小的网络架构

2. **训练不稳定**
   - 调整学习率
   - 增加目标网络更新频率
   - 检查奖励函数设计

3. **收敛缓慢**
   - 增加训练轮数
   - 调整ε-贪心参数
   - 优化网络架构

### 调试技巧

1. 使用快速训练配置进行调试
2. 启用详细日志记录
3. 定期保存检查点
4. 监控训练统计信息

## 扩展开发

### 添加新的网络架构

```python
class CustomNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_objectives=2):
        super().__init__()
        # 实现自定义网络架构
    
    def forward(self, state, preference):
        # 实现前向传播
        return q_values
```

### 添加新的偏好策略

```python
class CustomPreferenceManager(PreferenceManager):
    def sample_preference(self):
        # 实现自定义偏好采样策略
        return preference
```

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件
- 项目讨论区

## 更新日志

### v1.0.0
- 初始版本发布
- 实现完整的DDQN算法
- 支持多目标学习和动态权重
- 集成TPPT轨迹预测
- 提供多种配置选项 