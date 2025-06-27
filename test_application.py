#!/usr/bin/env python3
"""
测试修改后的application类
验证单任务结构和随机任务生成功能
"""

import sys
import os
sys.path.append('.')

import numpy as np
from Environments.application import application
from Environments.vehicles import vehicle
from Environments.env import Env

def test_application():
    """测试application类的基本功能"""
    print("=== 测试Application类 ===")
    
    # 设置随机种子以便复现结果
    np.random.seed(42)
    
    # 测试任务生成概率
    print("1. 测试任务生成概率:")
    apps = [application() for _ in range(100)]
    task_count = sum(1 for app in apps if app.has_task())
    print(f"   生成任务的应用数量: {task_count}/100")
    print(f"   任务生成概率: {task_count/100:.2f}")
    
    # 测试任务参数
    print("\n2. 测试任务参数:")
    app_with_task = None
    for app in apps:
        if app.has_task():
            app_with_task = app
            break
    
    if app_with_task:
        print(f"   任务数据量: {app_with_task.get_task_size()}")
        print(f"   任务计算密度: {app_with_task.get_task_density()}")
        print(f"   任务CPU周期数: {app_with_task.get_task_cycles()}")
        print(f"   服务实例大小: {app_with_task.instance}")
    
    # 测试任务重新生成
    print("\n3. 测试任务重新生成:")
    app = application()
    print(f"   初始任务: {app.task}")
    print(f"   是否有任务: {app.has_task()}")
    
    # 重新生成任务
    app.task = app.generate_task()
    print(f"   重新生成后: {app.task}")
    print(f"   是否有任务: {app.has_task()}")

def test_vehicle():
    """测试vehicle类与新的application的集成"""
    print("\n=== 测试Vehicle类 ===")
    
    np.random.seed(42)
    
    # 创建车辆
    veh = vehicle(0)
    print(f"1. 车辆初始任务: {veh.application.task}")
    print(f"   是否有任务: {veh.application.has_task()}")
    
    # 测试本地计算
    print("\n2. 测试本地计算:")
    delay, energy = veh.compute_local_task()
    print(f"   本地计算延迟: {delay:.6f}")
    print(f"   本地计算能耗: {energy:.6f}")
    
    # 测试任务卸载
    print("\n3. 测试任务卸载:")
    channel_rate = 1e6  # 1MB/s
    delay, energy = veh.offload_task(channel_rate)
    print(f"   卸载延迟: {delay:.6f}")
    print(f"   卸载能耗: {energy:.6f}")

def test_environment():
    """测试环境类与新的application的集成"""
    print("\n=== 测试Environment类 ===")
    
    np.random.seed(42)
    
    # 创建环境
    env = Env()
    print(f"1. 环境初始化完成")
    print(f"   车辆数量: {env.vehicle_number}")
    print(f"   基站数量: {env.cellular_number}")
    
    # 获取初始状态
    state = env.get_state()
    print(f"\n2. 初始状态形状: {state.shape}")
    print(f"   状态维度: {len(state)}")
    
    # 检查任务状态
    task_count = 0
    for i in range(env.vehicle_number):
        if env.vehicle_list[i].application.has_task():
            task_count += 1
    
    print(f"\n3. 任务统计:")
    print(f"   有任务的车辆数量: {task_count}/{env.vehicle_number}")
    print(f"   任务生成率: {task_count/env.vehicle_number:.2f}")
    
    # 测试一步执行
    print(f"\n4. 测试环境步进:")
    action = [0] * env.vehicle_number  # 所有车辆都选择基站0
    next_state, reward, vec_result, done = env.step(action)
    print(f"   奖励: {reward:.6f}")
    print(f"   是否结束: {done}")
    print(f"   下一状态形状: {next_state.shape}")

if __name__ == "__main__":
    test_application()
    test_vehicle()
    test_environment()
    print("\n=== 所有测试完成 ===") 