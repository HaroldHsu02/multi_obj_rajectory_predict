#!/usr/bin/env python3
"""
DDQN算法使用示例
展示如何快速开始使用基于改进DRL的动态权重算法
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Environments.env import Env
from DDQN_Framework import (
    VECEnvironmentAdapter, DDQNAgent, DDQNTrainer,
    get_config, validate_config, print_config_summary
)


def quick_start_example():
    """
    快速开始示例
    演示如何使用DDQN算法进行训练
    """
    print("=" * 60)
    print("DDQN算法快速开始示例")
    print("=" * 60)
    
    # 1. 获取配置
    config = get_config('fast_training')  # 使用快速训练配置
    print_config_summary(config)
    
    # 2. 验证配置
    if not validate_config(config):
        print("配置验证失败")
        return
    
    # 3. 设置环境
    print("\n步骤1: 设置环境")
    original_env = Env()
    env_adapter = VECEnvironmentAdapter(original_env, trajectory_predictor=None)
    env_adapter.set_reward_weights(
        config['environment']['delay_weight'],
        config['environment']['energy_weight']
    )
    
    print(f"环境设置完成:")
    print(f"  - 状态维度: {env_adapter.get_state_dim()}")
    print(f"  - 动作维度: {env_adapter.get_action_dim()}")
    print(f"  - 车辆数量: {env_adapter.get_vehicle_number()}")
    
    # 4. 设置智能体
    print("\n步骤2: 设置智能体")
    state_dim = env_adapter.get_state_dim()
    action_dim = env_adapter.get_action_dim()
    
    # 生成偏好空间
    preference_space_size = config['preference']['preference_space_size']
    preference_space = []
    step = 1.0 / (preference_space_size - 1) if preference_space_size > 1 else 1.0
    
    for i in range(preference_space_size):
        omega_d = i * step
        omega_e = 1.0 - omega_d
        preference = np.array([omega_d, omega_e])
        preference_space.append(preference)
    
    agent = DDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        preference_space=preference_space,
        config=config
    )
    
    print(f"智能体设置完成:")
    print(f"  - 网络架构: {'Dueling' if config['network']['use_dueling'] else 'Standard'}")
    print(f"  - 偏好空间大小: {len(preference_space)}")
    print(f"  - 设备: {agent.device}")
    
    # 5. 设置训练器
    print("\n步骤3: 设置训练器")
    trainer = DDQNTrainer(env_adapter, agent, config)
    print("训练器设置完成")
    
    # 6. 开始训练
    print("\n步骤4: 开始训练")
    print("注意: 这是一个演示，只训练少量episode")
    
    # 修改配置以进行快速演示
    demo_config = config.copy()
    demo_config['training']['max_episodes'] = 10  # 只训练10个episode
    demo_config['training']['horizon_steps'] = 50  # 减少步数
    demo_config['training']['eval_interval'] = 5   # 更频繁的评估
    
    trainer.config = demo_config
    
    try:
        training_result = trainer.train(max_episodes=10)
        print(f"\n训练完成!")
        print(f"总训练轮数: {training_result['total_episodes']}")
        print(f"最终平均奖励: {training_result['final_avg_reward']:.3f}")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        print("这可能是由于环境或网络配置问题导致的")
    
    print("\n快速开始示例完成!")


def preference_example():
    """
    偏好管理示例
    演示如何使用偏好管理器
    """
    print("\n" + "=" * 60)
    print("偏好管理示例")
    print("=" * 60)
    
    from DDQN_Framework import PreferenceManager
    
    # 创建偏好管理器
    preference_config = {
        'preference_space_size': 6,
        'non_dominated_sorting': True,
        'validation_enabled': True
    }
    
    preference_manager = PreferenceManager(preference_config)
    
    # 显示偏好空间
    preference_space = preference_manager.get_preference_space()
    print("候选偏好空间:")
    for i, pref in enumerate(preference_space):
        print(f"  {i}: [ωD={pref[0]:.1f}, ωE={pref[1]:.1f}]")
    
    # 采样偏好
    print("\n偏好采样示例:")
    for i in range(5):
        preference = preference_manager.sample_preference()
        print(f"  采样 {i+1}: [ωD={preference[0]:.1f}, ωE={preference[1]:.1f}]")
    
    # 添加遇到偏好
    print("\n添加遇到偏好:")
    test_preferences = [
        np.array([1.0, 0.0]),  # 只关注延迟
        np.array([0.0, 1.0]),  # 只关注能耗
        np.array([0.5, 0.5]),  # 平衡偏好
    ]
    
    for pref in test_preferences:
        preference_manager.add_encountered_preference(pref)
        print(f"  添加: [ωD={pref[0]:.1f}, ωE={pref[1]:.1f}]")
    
    # 选择遇到偏好
    print("\n选择遇到偏好:")
    for i in range(3):
        selected_pref = preference_manager.select_encountered_preference()
        print(f"  选择 {i+1}: [ωD={selected_pref[0]:.1f}, ωE={selected_pref[1]:.1f}]")


def experience_buffer_example():
    """
    经验缓冲池示例
    演示如何使用多目标经验缓冲池
    """
    print("\n" + "=" * 60)
    print("经验缓冲池示例")
    print("=" * 60)
    
    from DDQN_Framework import MultiObjectiveExperienceBuffer
    
    # 创建经验缓冲池
    buffer = MultiObjectiveExperienceBuffer(
        capacity=1000,
        state_dim=10,
        action_dim=5
    )
    
    # 存储经验
    print("存储经验示例:")
    for i in range(5):
        state = np.random.randn(10)
        action = np.random.randint(0, 5, size=1)
        reward_vector = np.array([-1.0, -0.5])  # [延迟奖励, 能耗奖励]
        next_state = np.random.randn(10)
        done = False
        preference = np.array([0.7, 0.3])  # [ωD, ωE]
        
        success = buffer.store(state, action, reward_vector, next_state, done, preference)
        print(f"  经验 {i+1}: {'成功' if success else '失败'}")
    
    # 采样批次
    print(f"\n缓冲池大小: {buffer.get_buffer_size()}")
    
    if buffer.get_buffer_size() >= 3:
        batch = buffer.sample_batch_numpy(3)
        states, actions, rewards, next_states, dones, preferences = batch
        
        print("采样批次示例:")
        print(f"  状态形状: {states.shape}")
        print(f"  动作形状: {actions.shape}")
        print(f"  奖励形状: {rewards.shape}")
        print(f"  偏好形状: {preferences.shape}")
        
        # 显示统计信息
        stats = buffer.get_statistics()
        print(f"\n缓冲池统计:")
        print(f"  总经验数: {stats['total_experiences']}")
        print(f"  有效经验数: {stats['valid_experiences']}")
        print(f"  有效性率: {stats['validity_rate']:.2f}")
        print(f"  平均延迟奖励: {stats['avg_delay_reward']:.3f}")
        print(f"  平均能耗奖励: {stats['avg_energy_reward']:.3f}")


def network_example():
    """
    神经网络示例
    演示如何使用多目标神经网络
    """
    print("\n" + "=" * 60)
    print("神经网络示例")
    print("=" * 60)
    
    import torch
    from DDQN_Framework import MultiObjectiveQNetwork, DuelingMultiObjectiveQNetwork
    
    # 创建标准多目标网络
    print("标准多目标网络:")
    standard_network = MultiObjectiveQNetwork(
        state_dim=10,
        action_dim=5,
        hidden_dims=[64, 32],
        num_objectives=2
    )
    
    # 测试前向传播
    state = torch.randn(2, 10)  # 批次大小为2
    preference = torch.tensor([[0.7, 0.3], [0.3, 0.7]])  # 两个不同的偏好
    
    with torch.no_grad():
        q_values = standard_network(state, preference)
        print(f"  输入状态形状: {state.shape}")
        print(f"  输入偏好形状: {preference.shape}")
        print(f"  输出Q值形状: {q_values.shape}")
        print(f"  Q值范围: [{q_values.min():.3f}, {q_values.max():.3f}]")
    
    # 创建Dueling多目标网络
    print("\nDueling多目标网络:")
    dueling_network = DuelingMultiObjectiveQNetwork(
        state_dim=10,
        action_dim=5,
        hidden_dims=[64, 32],
        num_objectives=2
    )
    
    # 测试前向传播
    with torch.no_grad():
        q_values = dueling_network(state, preference)
        print(f"  输入状态形状: {state.shape}")
        print(f"  输入偏好形状: {preference.shape}")
        print(f"  输出Q值形状: {q_values.shape}")
        print(f"  Q值范围: [{q_values.min():.3f}, {q_values.max():.3f}]")
    
    # 计算网络参数数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n网络参数数量:")
    print(f"  标准网络: {count_parameters(standard_network):,}")
    print(f"  Dueling网络: {count_parameters(dueling_network):,}")


def main():
    """主函数"""
    print("DDQN算法使用示例")
    print("本示例将演示DDQN算法的各个组件")
    
    try:
        # 快速开始示例
        quick_start_example()
        
        # 偏好管理示例
        preference_example()
        
        # 经验缓冲池示例
        experience_buffer_example()
        
        # 神经网络示例
        network_example()
        
        print("\n" + "=" * 60)
        print("所有示例完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"示例执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 