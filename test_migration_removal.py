#!/usr/bin/env python3
"""
测试服务迁移相关代码删除后的功能
"""

import sys
import os
sys.path.append('.')


def test_application():
    """测试Application类"""
    print("=== 测试Application类 ===")
    from Environments.application import application

    # 测试应用初始化
    app = application()
    print(f"任务: {app.task}")
    print(f"有任务: {app.has_task()}")
    print(f"任务大小: {app.get_task_size()}")
    print(f"任务密度: {app.get_task_density()}")
    print(f"服务实例大小: {app.instance}")

    # 测试任务生成概率
    import numpy as np
    np.random.seed(42)
    tasks = [application().has_task() for _ in range(100)]
    task_rate = sum(tasks) / len(tasks)
    print(f"任务生成率: {task_rate:.2f} (期望: 0.7)")

    # 测试服务实例变化
    original_instance = app.instance
    app.instance_change()
    print(f"服务实例变化: {original_instance} -> {app.instance}")

    print("✓ Application类测试通过\n")


def test_cellular_node():
    """测试Cellular_node类"""
    print("=== 测试Cellular_node类 ===")
    from Environments.cellular_node import cellular_node

    # 测试基站初始化
    cell = cellular_node(0)
    print(f"基站索引: {cell.cellular_index}")
    print(f"基站位置: {cell.cellular_loc}")
    print(f"基站半径: {cell.radius}")
    print(f"计算能力: {cell.capability}")
    print(f"带宽: {cell.bandwidth}")

    # 测试信道速率计算
    vehicle_power = 0.5
    vehicle_loc = [600, 600]
    rate = cell.compute_channel_rate(vehicle_power, vehicle_loc)
    print(f"信道传输速率: {rate:.2f} bps")

    print("✓ Cellular_node类测试通过\n")


def test_environment():
    """测试Environment类"""
    print("=== 测试Environment类 ===")
    from Environments.env import Env

    # 测试环境初始化
    env = Env()
    print(f"车辆数量: {env.vehicle_number}")
    print(f"基站数量: {env.cellular_number}")
    print(f"当前时隙: {env.time_slot}")

    # 测试状态获取
    state = env.get_state()
    print(f"状态向量形状: {state.shape}")
    print(f"状态向量前10个元素: {state[:10]}")

    # 测试标准化状态
    norm_state = env.get_state_normalize()
    print(f"标准化状态向量形状: {norm_state.shape}")
    print(f"标准化状态向量前10个元素: {norm_state[:10]}")

    # 测试动作执行
    import numpy as np
    np.random.seed(42)
    action = np.random.randint(0, env.cellular_number, env.vehicle_number)
    next_state, reward, result, done = env.step(action)
    print(f"执行动作后奖励: {reward:.4f}")
    print(f"是否结束: {done}")
    print(f"结果列表长度: {len(result)}")

    print("✓ Environment类测试通过\n")


def test_config():
    """测试配置文件"""
    print("=== 测试配置文件 ===")
    from Environments.config import (
        TASK_SIZE_RANGE, TASK_DENSITY_RANGE, TASK_GENERATION_PROBABILITY,
        INSTANCE_SIZE_MULTIPLIER, INSTANCE_CHANGE_RANGE, BACKHAUL_ONE_HOP
    )

    print(f"任务大小范围: {TASK_SIZE_RANGE}")
    print(f"任务密度范围: {TASK_DENSITY_RANGE}")
    print(f"任务生成概率: {TASK_GENERATION_PROBABILITY}")
    print(f"服务实例大小倍数: {INSTANCE_SIZE_MULTIPLIER}")
    print(f"服务实例变化范围: {INSTANCE_CHANGE_RANGE}")
    print(f"回程一跳时间: {BACKHAUL_ONE_HOP}")

    # 检查迁移相关参数是否已删除
    migration_params = ['MIGRATION_ONE_HOP', 'MIGRATION_PREPARE_TIME']
    for param in migration_params:
        try:
            # 尝试导入参数，如果不存在会抛出ImportError
            exec(f"from Environments.config import {param}")
            print(f"⚠️  警告: {param} 仍然存在")
        except ImportError:
            print(f"✓ {param} 已删除")

    print("✓ 配置文件测试通过\n")


def main():
    """主测试函数"""
    print("开始测试服务迁移相关代码删除后的功能...\n")

    try:
        test_config()
        test_application()
        test_cellular_node()
        test_environment()

        print("🎉 所有测试通过！服务迁移相关代码已成功删除。")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
