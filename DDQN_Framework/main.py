#!/usr/bin/env python3
"""
DDQN算法主程序入口
实现基于改进DRL的动态权重算法的完整运行流程
"""

from DDQN_Framework.config import get_config, validate_config, print_config_summary
from DDQN_Framework.Training.trainer import DDQNTrainer
from DDQN_Framework.Models.ddqn_agent import DDQNAgent
from DDQN_Framework.Adapters.vec_environment_adapter import VECEnvironmentAdapter
from T_Pattern_Tree.predictor import TrajectoryPredictor
from Environments.env import Env
import sys
import os
import argparse
import numpy as np
import torch
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def setup_environment(config: Dict[str, Any]) -> VECEnvironmentAdapter:
    """
    设置环境
    Args:
        config: 配置字典
    Returns:
        环境适配器
    """
    print("正在初始化环境...")

    # 创建原始环境
    original_env = Env()

    # 创建TPPT预测器（如果启用）
    trajectory_predictor = None
    if config['tppt']['enabled']:
        try:
            print("正在初始化TPPT轨迹预测器...")
            trajectory_predictor = TrajectoryPredictor(
                rsu_locations=None,  # 这里需要根据实际RSU位置设置
                config=config['tppt']
            )
            print("TPPT轨迹预测器初始化成功")
        except Exception as e:
            print(f"TPPT初始化失败: {e}")
            print("将禁用TPPT预测功能")
            trajectory_predictor = None

    # 创建环境适配器
    env_adapter = VECEnvironmentAdapter(original_env, trajectory_predictor)

    # 设置奖励权重
    env_adapter.set_reward_weights(
        config['environment']['delay_weight'],
        config['environment']['energy_weight']
    )

    print(f"环境初始化完成:")
    print(f"  - 车辆数量: {env_adapter.get_vehicle_number()}")
    print(f"  - 动作空间: {env_adapter.get_action_dim()}")
    print(f"  - 状态维度: {env_adapter.get_state_dim()}")
    print(f"  - TPPT启用: {trajectory_predictor is not None}")

    return env_adapter


def setup_agent(env_adapter: VECEnvironmentAdapter, config: Dict[str, Any]) -> DDQNAgent:
    """
    设置DDQN智能体
    Args:
        env_adapter: 环境适配器
        config: 配置字典
    Returns:
        DDQN智能体
    """
    print("正在初始化DDQN智能体...")

    # 获取环境参数
    state_dim = env_adapter.get_state_dim()
    action_dim = env_adapter.get_action_dim()

    # 生成偏好空间
    preference_space_size = config['preference']['preference_space_size']
    preference_space = []
    step = 1.0 / (preference_space_size -
                  1) if preference_space_size > 1 else 1.0

    for i in range(preference_space_size):
        omega_d = i * step  # 延迟偏好权重
        omega_e = 1.0 - omega_d  # 能耗偏好权重
        preference = np.array([omega_d, omega_e])
        preference_space.append(preference)

    # 创建智能体
    agent = DDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        preference_space=preference_space,
        config=config
    )

    print(f"智能体初始化完成:")
    print(f"  - 状态维度: {state_dim}")
    print(f"  - 动作维度: {action_dim}")
    print(f"  - 偏好空间大小: {len(preference_space)}")
    print(
        f"  - 网络架构: {'Dueling' if config['network']['use_dueling'] else 'Standard'}")
    print(f"  - 设备: {agent.device}")

    return agent


def setup_trainer(env_adapter: VECEnvironmentAdapter, agent: DDQNAgent,
                  config: Dict[str, Any]) -> DDQNTrainer:
    """
    设置训练器
    Args:
        env_adapter: 环境适配器
        agent: DDQN智能体
        config: 配置字典
    Returns:
        训练器
    """
    print("正在初始化训练器...")

    trainer = DDQNTrainer(env_adapter, agent, config)

    print("训练器初始化完成")
    return trainer


def run_training(trainer: DDQNTrainer, config: Dict[str, Any],
                 max_episodes: int = None) -> Dict[str, Any]:
    """
    运行训练
    Args:
        trainer: 训练器
        config: 配置字典
        max_episodes: 最大训练轮数
    Returns:
        训练结果
    """
    print("开始训练...")

    # 开始训练
    training_result = trainer.train(max_episodes=max_episodes)

    print("训练完成!")
    print(f"总训练轮数: {training_result['total_episodes']}")
    print(f"总步数: {training_result['total_steps']}")
    print(f"训练时间: {training_result['training_time']:.2f}秒")
    print(f"最终平均奖励: {training_result['final_avg_reward']:.3f}")
    print(f"遇到偏好数量: {training_result['encountered_preferences']}")

    return training_result


def run_evaluation(trainer: DDQNTrainer, eval_episodes: int = 50) -> Dict[str, float]:
    """
    运行评估
    Args:
        trainer: 训练器
        eval_episodes: 评估轮数
    Returns:
        评估结果
    """
    print(f"开始评估 ({eval_episodes} episodes)...")

    eval_result = trainer.evaluate_performance(eval_episodes=eval_episodes)

    print("评估完成!")
    print(
        f"平均奖励: {eval_result['avg_reward']:.3f} ± {eval_result['std_reward']:.3f}")
    print(f"平均延迟奖励: {eval_result['avg_delay_reward']:.3f}")
    print(f"平均能耗奖励: {eval_result['avg_energy_reward']:.3f}")
    print(
        f"奖励范围: [{eval_result['min_reward']:.3f}, {eval_result['max_reward']:.3f}]")

    return eval_result


def save_results(trainer: DDQNTrainer, training_result: Dict[str, Any],
                 eval_result: Dict[str, float], config: Dict[str, Any]):
    """
    保存结果
    Args:
        trainer: 训练器
        training_result: 训练结果
        eval_result: 评估结果
        config: 配置字典
    """
    print("正在保存结果...")

    # 创建结果目录
    import os
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/ddqn_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # 保存最终模型
    model_path = os.path.join(results_dir, "final_model.pth")
    trainer.agent.save_model(model_path)

    # 保存训练摘要
    training_summary = trainer.get_training_summary()
    training_summary['training_result'] = training_result
    training_summary['eval_result'] = eval_result
    training_summary['config'] = config

    import pickle
    summary_path = os.path.join(results_dir, "training_summary.pkl")
    with open(summary_path, 'wb') as f:
        pickle.dump(training_summary, f)

    # 保存配置
    import json
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"结果已保存到: {results_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DDQN算法训练程序')
    parser.add_argument('--config', type=str, default='default',
                        help='配置名称 (default, fast_training, high_precision, delay_focused, energy_focused)')
    parser.add_argument('--episodes', type=int, default=None,
                        help='训练轮数 (覆盖配置文件中的设置)')
    parser.add_argument('--eval_episodes', type=int, default=50,
                        help='评估轮数')
    parser.add_argument('--no_training', action='store_true',
                        help='跳过训练，只进行评估')
    parser.add_argument('--load_checkpoint', type=int, default=None,
                        help='加载指定episode的检查点')

    args = parser.parse_args()

    print("=" * 60)
    print("基于改进DRL的动态权重算法 - DDQN实现")
    print("=" * 60)

    # 获取配置
    config = get_config(args.config)
    print_config_summary(config)

    # 验证配置
    if not validate_config(config):
        print("配置验证失败，程序退出")
        return

    # 设置随机种子
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    try:
        # 设置环境
        env_adapter = setup_environment(config)

        # 设置智能体
        agent = setup_agent(env_adapter, config)

        # 设置训练器
        trainer = setup_trainer(env_adapter, agent, config)

        # 加载检查点（如果指定）
        if args.load_checkpoint is not None:
            print(f"加载检查点: episode {args.load_checkpoint}")
            trainer.load_checkpoint(args.load_checkpoint)

        training_result = None

        # 运行训练（除非指定跳过）
        if not args.no_training:
            training_result = run_training(trainer, config, args.episodes)

        # 运行评估
        eval_result = run_evaluation(trainer, args.eval_episodes)

        # 保存结果
        if training_result is not None:
            save_results(trainer, training_result, eval_result, config)

        print("=" * 60)
        print("程序执行完成!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
